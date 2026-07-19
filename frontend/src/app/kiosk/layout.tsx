import type { Metadata } from "next";
import "./kiosk.css";

export const metadata: Metadata = {
  title: "İTÜ Tercih Danışmanı",
  description: "İTÜ yapay zekâ tercih danışmanı — gerçek zamanlı sesli görüşme.",
};

export default function KioskLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return <div className="kiosk-theme">{children}</div>;
}
