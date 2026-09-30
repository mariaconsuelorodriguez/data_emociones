"""Genera la ficha informativa y la infografía de la ruta de articulación (PNG + PDF).

Uso: python3 crear_piezas.py
"""
from playwright.sync_api import sync_playwright

from contenido import (CARGOS, CREDITOS, CURSOS_ARTICULACION, POR_QUE_UNAD, PROGRAMA, QUE_HACE, RUTA,
                       SECTORES, TITULO)
from estilos import CHROME, OUT, doc

TOTAL = sum(c for _, c, _ in CURSOS_ARTICULACION)

FICHA_CSS = """
.pg{width:1240px;height:1754px;background:var(--paper);position:relative;overflow:hidden}
.hd{background:linear-gradient(135deg,var(--navy),var(--navy2));color:#fff;padding:70px 80px 60px;position:relative}
.hd .bar{position:absolute;left:0;top:0;bottom:0;width:16px;background:var(--gold)}
.tag{display:inline-block;background:var(--gold);color:var(--navy);font-weight:800;font-size:20px;padding:8px 20px;border-radius:30px}
h1{font-size:60px;font-weight:900;line-height:1.05;margin-top:22px}
.gold{color:var(--gold)} .teal{color:var(--teal)}
.facts{display:flex;gap:16px;margin-top:30px}
.fact{background:rgba(255,255,255,.1);border-radius:16px;padding:14px 22px}
.fact b{display:block;font-size:26px;color:var(--gold);font-weight:900}.fact span{font-size:16px;font-weight:600}
.body{padding:44px 80px 0}
h2{font-size:28px;color:var(--navy);font-weight:900;margin:0 0 18px;display:flex;align-items:center;gap:12px}
h2:before{content:'';width:10px;height:30px;background:var(--gold);border-radius:4px}
.sec{margin-bottom:38px}
.row{display:flex;gap:18px}
.card{flex:1;background:#fff;border-radius:18px;padding:22px 22px;border:1.5px solid #DDE3EE}
.card .emoji{font-size:34px}.card h3{font-size:19px;color:var(--navy);font-weight:800;margin:10px 0 6px}
.card p{font-size:15.5px;line-height:1.4;color:var(--ink)}
.jobs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.job{background:#fff;border:1.5px solid #DDE3EE;border-radius:14px;padding:13px 16px;font-size:16.5px;font-weight:700;color:var(--navy);display:flex;gap:12px;align-items:center}
.chips{display:flex;flex-wrap:wrap;gap:9px;margin-top:16px}
.chip{background:var(--navy);color:#fff;font-weight:700;font-size:14.5px;padding:7px 16px;border-radius:20px}
.course{flex:1;background:var(--navy);color:#fff;border-radius:18px;padding:22px}
.course b{font-size:44px;color:var(--gold);font-weight:900}.course small{font-size:16px;color:var(--gold);font-weight:800}
.course h3{font-size:19px;font-weight:800;margin:6px 0}.course p{font-size:14.5px;opacity:.9;line-height:1.35}
.ft{position:absolute;left:0;right:0;bottom:0;background:var(--gold);color:var(--navy);padding:22px 80px;display:flex;justify-content:space-between;align-items:center;font-weight:800;font-size:20px}
.note{font-size:14px;color:var(--muted);margin-top:12px}
"""


def ficha():
    que = "".join(f"<div class='card'><div class='emoji'>{e}</div><h3>{t}</h3><p>{d[0].upper() + d[1:]}</p></div>"
                  for e, t, d in QUE_HACE)
    jobs = "".join(f"<div class='job'><span class='emoji' style='font-size:24px'>{e}</span>{t}</div>" for e, t in CARGOS)
    chips = "".join(f"<span class='chip'>{s}</span>" for s in SECTORES)
    cursos = "".join(f"<div class='course'><b>{c}</b> <small>créditos</small><h3>{n}</h3><p>{d}</p></div>"
                     for n, c, d in CURSOS_ARTICULACION)
    return doc(f"""<div class='pg'>
    <div class='hd'><div class='bar'></div>
      <span class='tag'>Ficha informativa · Articulación SINEP – ECBTI</span>
      <h1 style='font-size:56px'>Ingeniería en <span class='gold'>Inteligencia Artificial</span><br>y <span class='teal'>Ciencia de Datos</span></h1>
      <div class='facts'>
        <div class='fact'><b>Pregrado</b><span>Profesional universitario</span></div>
        <div class='fact'><b>Virtual</b><span>Modalidad</span></div>
        <div class='fact'><b>{CREDITOS}</b><span>Créditos académicos</span></div>
        <div class='fact'><b>ECBTI</b><span>Escuela de Ciencias Básicas, Tecnología e Ingeniería</span></div>
      </div>
      <div style='margin-top:20px;font-size:18px;font-weight:600'>Título: {TITULO}</div>
    </div>
    <div class='body'>
      <div class='sec'><h2>Perfil: ¿qué hace el profesional?</h2><div class='row'>{que}</div></div>
      <div class='sec'><h2>Campo laboral: ¿en qué puedes trabajar?</h2><div class='jobs'>{jobs}</div><div class='chips'>{chips}</div></div>
      <div class='sec'><h2>Cursos de articulación con el SINEP</h2><div class='row'>{cursos}</div>
        <div style='margin-top:16px;font-size:18px;font-weight:700;color:var(--navy)'>✔ Hasta {TOTAL} créditos reconocidos al ingresar al programa · ✔ Ahorras tiempo y dinero · ✔ Continuidad académica</div>
      </div>
    </div>
    <div class='ft'><span>Del SINEP a la Ingeniería</span><span>UNAD · Más UNAD, más equidad</span></div>
    </div>""", FICHA_CSS)


INFO_CSS = """
.pg{width:1080px;height:1920px;background:linear-gradient(180deg,var(--navy) 0%,var(--navy2) 100%);color:#fff;position:relative;overflow:hidden;padding:80px 80px 0}
.grid-bg{position:absolute;inset:0}
.tag{display:inline-block;background:var(--gold);color:var(--navy);font-weight:800;font-size:22px;padding:9px 22px;border-radius:30px}
h1{font-size:68px;font-weight:900;line-height:1.05;margin-top:24px}
.gold{color:var(--gold)} .teal{color:var(--teal)}
.sub{font-size:24px;font-weight:600;margin-top:16px;opacity:.9}
.steps{position:relative;margin-top:50px}
.line{position:absolute;left:52px;top:40px;bottom:40px;width:6px;background:repeating-linear-gradient(180deg,var(--gold) 0 18px,transparent 18px 30px)}
.st{display:flex;gap:30px;align-items:center;margin-bottom:34px;position:relative}
.n{flex:none;width:110px;height:110px;border-radius:50%;background:var(--gold);color:var(--navy);font-size:52px;font-weight:900;display:flex;align-items:center;justify-content:center;box-shadow:0 0 0 10px rgba(253,185,19,.18)}
.bx{flex:1;background:rgba(255,255,255,.09);border:2px solid rgba(255,255,255,.15);border-radius:24px;padding:22px 28px}
.bx h3{font-size:32px;font-weight:900;color:var(--gold)}.bx p{font-size:21px;line-height:1.4;margin-top:6px;font-weight:500}
.why{background:#fff;color:var(--navy);border-radius:28px;padding:32px 34px;margin-top:30px}
.why h2{font-size:30px;font-weight:900}
.wg{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;margin-top:14px}
.wg div{font-size:19px;font-weight:700;display:flex;gap:10px;align-items:center}
.ft{position:absolute;left:0;right:0;bottom:0;background:var(--gold);color:var(--navy);padding:24px 80px;display:flex;justify-content:space-between;font-weight:900;font-size:24px}
"""


def infografia():
    steps = "".join(f"<div class='st'><div class='n'>{n}</div><div class='bx'><h3>{t}</h3><p>{d}</p></div></div>"
                    for n, t, d in RUTA)
    why = "".join(f"<div><span class='emoji'>{e}</span>{t}</div>" for e, t, _ in POR_QUE_UNAD)
    return doc(f"""<div class='pg'><div class='grid-bg'></div><div style='position:relative'>
      <span class='tag'>Ruta de articulación SINEP → UNAD</span>
      <h1>Del SINEP a la <span class='gold'>Ingeniería</span><br>en <span class='teal'>IA y Ciencia de Datos</span></h1>
      <div class='sub'>Tu ruta hacia la educación superior, paso a paso</div>
      <div class='steps'><div class='line'></div>{steps}</div>
      <div class='why'><h2>¿Por qué estudiar en la UNAD?</h2><div class='wg'>{why}<div><span class='emoji'>✅</span>Hasta {TOTAL} créditos adelantados</div></div></div>
    </div><div class='ft'><span>UNAD · ECBTI</span><span>¡Empieza hoy!</span></div></div>""", INFO_CSS)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        for name, html, w, h in [("03_ficha_informativa_IA_CD", ficha(), 1240, 1754),
                                 ("04_infografia_ruta_articulacion", infografia(), 1080, 1920)]:
            pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            pg.set_content(html, wait_until="load"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}.png"))
            pg.pdf(path=str(OUT / f"{name}.pdf"), width=f"{w}px", height=f"{h}px", print_background=True,
                   margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            pg.close()
        b.close()


if __name__ == "__main__":
    main()
