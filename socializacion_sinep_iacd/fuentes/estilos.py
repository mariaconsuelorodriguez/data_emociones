"""Estilos y utilidades de renderizado HTML -> PNG/PDF con Chromium (Playwright)."""
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

CSS = """
:root{--navy:#0B2A5B;--navy2:#123E7C;--gold:#FDB913;--teal:#19B3B1;--ink:#1B2433;--muted:#5B6678;--paper:#F6F8FC;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Montserrat','DejaVu Sans',sans-serif;color:var(--ink);background:#fff}
.emoji{font-family:'Noto Color Emoji';}
.grid-bg{background-image:radial-gradient(rgba(255,255,255,.10) 1.5px,transparent 1.5px);background-size:36px 36px}
"""


def render(pages, width, height, pdf_path=None):
    """pages: lista de (html, png_path). Si pdf_path, además exporta un PDF multipágina."""
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        for html, png in pages:
            pg.set_content(html, wait_until="load")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(png), full_page=False)
        if pdf_path:
            full = pages[0][0] if len(pages) == 1 else None
            pg.set_content(full, wait_until="load")
            pg.pdf(path=str(pdf_path), width=f"{width}px", height=f"{height}px",
                   print_background=True, page_ranges="1")
        b.close()


def doc(body, extra_css=""):
    return f"<!doctype html><html lang='es'><head><meta charset='utf-8'><style>{CSS}{extra_css}</style></head><body>{body}</body></html>"


OUT = Path(__file__).resolve().parent.parent
