// PCM16 capture worklet for the İTÜ kiosk.
//
// Takes mono mic audio at the AudioContext's native rate (commonly 48 kHz),
// streaming-resamples it to a target rate (16 kHz for Gemini Live) with linear
// interpolation across process() blocks, converts to little-endian Int16, and
// posts the ArrayBuffer to the main thread (which forwards it over the WS).
//
// Mirrors the linear-interp fallback of resample_mono_float32 in
// backend/orchestrator/audio_helpers.py so uplink/downlink stay symmetric.

class Pcm16CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    // Batch outgoing audio into ~20ms frames (320 samples @16k) instead of one
    // message per 128-sample process() block — far fewer WS/API messages.
    this.frameSamples = opts.frameSamples || Math.round(this.targetRate / 50);
    this._out = [];                   // pending resampled samples
    this._tail = new Float32Array(0); // unconsumed input samples
    this._pos = 0;                    // fractional read index into (_tail + block)
  }

  _concat(a, b) {
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true;
    }
    const block = input[0]; // mono channel 0
    const buf = this._concat(this._tail, block);
    const ratio = sampleRate / this.targetRate; // sampleRate is the worklet global

    let pos = this._pos;
    while (Math.floor(pos) + 1 < buf.length) {
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      this._out.push(buf[i0] * (1 - frac) + buf[i0 + 1] * frac);
      pos += ratio;
    }

    // Keep the tail from the last integer index we still need for interpolation.
    const keepFrom = Math.floor(pos);
    this._tail = buf.subarray(Math.min(keepFrom, buf.length));
    this._pos = pos - keepFrom;

    // Emit complete ~20ms frames.
    while (this._out.length >= this.frameSamples) {
      const chunk = this._out.splice(0, this.frameSamples);
      const pcm = new Int16Array(this.frameSamples);
      for (let i = 0; i < chunk.length; i++) {
        let s = chunk[i];
        s = s < -1 ? -1 : s > 1 ? 1 : s;
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      // Transfer the buffer to avoid a copy.
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm16-capture-processor", Pcm16CaptureProcessor);
