"""Official İTÜ Akademi professor-search parsing."""

from professor_search import parse_profile, parse_search_people


def test_parse_search_people_extracts_official_card():
    page = """
      <div class="profil-image">
        <img src='http://akademi.itu.edu.tr/PublicPhoto/sanem.jpg' alt=''>
      </div>
      <span class="title">Prof. Dr.</span>
      <span class="name">Sanem Sarıel Uzer</span>
      <div data-ajax-href="/summary/person/sariel/summary"></div>
    """
    assert parse_search_people(page) == [
        {
            "name": "Sanem Sarıel Uzer",
            "title": "Prof. Dr.",
            "slug": "sariel",
            "image_url": "https://akademi.itu.edu.tr/PublicPhoto/sanem.jpg",
            "profile_url": "https://akademi.itu.edu.tr/tr/sariel/",
        }
    ]


def test_parse_profile_extracts_research_metadata():
    page = """
      <span class="title">Çalışma Alanları</span>
      <span class="desc">: <a>Robotik</a>, <a>Yapay Zeka</a></span>
      <span class="title">Çalıştığı Birim</span>
      <span class="desc">: <a>Bilgisayar ve Bilişim</a>, Yapay Zeka Ve Veri Mühendisliği</span>
      <h3><span>HAKKINDA</span></h3>
      <p>Robot öğrenmesi üzerine çalışır. Laboratuvar yürütücüsüdür.</p>
      <div class="social-list"></div>
    """
    assert parse_profile(page) == {
        "work_areas": "Robotik, Yapay Zeka",
        "department": "Bilgisayar ve Bilişim, Yapay Zeka Ve Veri Mühendisliği",
        "summary": "Robot öğrenmesi üzerine çalışır. Laboratuvar yürütücüsüdür.",
    }
