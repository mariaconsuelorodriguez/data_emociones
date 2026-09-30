"""Genera el video (≤ 2 min) a partir de escenas HTML + narración sintética (Piper TTS).

Uso: python3 crear_video.py --voz /ruta/voz.onnx
Salida: ../video/video_IA_CD_SINEP.mp4 y ../video/subtitulos.srt
"""
import argparse
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg

from contenido import CARGOS, CURSOS_ARTICULACION, JORNADA, PROGRAMA, QUE_HACE, SECTORES
from estilos import OUT, doc, render

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1920, 1080

SCENE_CSS = """
.s{width:1920px;height:1080px;position:relative;overflow:hidden;background:linear-gradient(135deg,var(--navy) 0%,var(--navy2) 100%);color:#fff;padding:90px 140px 120px;display:flex;flex-direction:column;justify-content:center;align-items:flex-start}
.tag{display:inline-block;background:var(--gold);color:var(--navy);font-weight:800;font-size:30px;padding:10px 26px;border-radius:40px;letter-spacing:.5px}
h1{font-size:92px;font-weight:900;line-height:1.05;margin-top:34px}
h2{font-size:70px;font-weight:800;line-height:1.1;margin-top:28px}
.gold{color:var(--gold)} .teal{color:var(--teal)}
.foot{position:absolute;left:140px;right:140px;bottom:60px;display:flex;justify-content:space-between;font-size:26px;font-weight:600;opacity:.85}
.bar{position:absolute;left:0;top:0;width:18px;height:100%;background:var(--gold)}
.cards{display:flex;gap:40px;margin-top:70px;width:100%}
.card{flex:1;background:rgba(255,255,255,.08);border:2px solid rgba(255,255,255,.16);border-radius:30px;padding:46px 40px}
.card .emoji{font-size:84px}
.card h3{font-size:40px;font-weight:800;margin:26px 0 14px;color:var(--gold)}
.card p{font-size:30px;line-height:1.4;font-weight:500}
.circle{position:absolute;border-radius:50%;border:3px solid rgba(253,185,19,.35)}
"""


def foot():
    return "<div class='foot'><span>UNAD · ECBTI</span><span>Del SINEP a la Ingeniería</span></div><div class='bar'></div>"


def escenas():
    cursos_html = "".join(
        f"<div class='card' style='padding:40px'><div style='font-size:92px;font-weight:900;color:var(--gold)'>{c}<span style='font-size:36px'> créditos</span></div>"
        f"<h3 style='color:#fff;font-size:40px'>{n}</h3><p>{d}</p></div>"
        for n, c, d in CURSOS_ARTICULACION)
    total = sum(c for _, c, _ in CURSOS_ARTICULACION)
    cargos_html = "".join(
        f"<div style='display:flex;align-items:center;gap:22px;background:rgba(255,255,255,.08);border-radius:22px;padding:22px 28px'>"
        f"<span class='emoji' style='font-size:52px'>{e}</span><span style='font-size:32px;font-weight:700'>{t}</span></div>"
        for e, t in CARGOS)
    sectores_html = "".join(
        f"<span style='background:var(--gold);color:var(--navy);font-weight:800;font-size:28px;padding:10px 24px;border-radius:30px'>{s}</span>"
        for s in SECTORES)
    que_html = "".join(
        f"<div class='card'><div class='emoji'>{e}</div><h3>{t}</h3><p>{d}</p></div>" for e, t, d in QUE_HACE)

    return [
        # 1. Gancho
        ("¿Alguna vez te has preguntado cómo tu celular reconoce tu cara, cómo una aplicación sabe qué canción "
         "recomendarte o cómo un mapa encuentra la ruta más rápida? Detrás de todo eso hay inteligencia artificial y datos.",
         f"""<div class='s grid-bg'>
         <div class='circle' style='width:900px;height:900px;right:-250px;top:-200px'></div>
         <div class='circle' style='width:600px;height:600px;right:-100px;top:-50px;border-color:rgba(25,179,177,.4)'></div>
         <span class='tag'>Del SINEP a la Ingeniería</span>
         <h1 style='margin-top:60px;font-size:110px'>¿Cómo lo hace<br><span class='gold'>tu celular?</span></h1>
         <div style='display:flex;gap:60px;margin-top:90px'>
           {''.join(f"<div style='text-align:center'><div class='emoji' style='font-size:120px'>{e}</div><div style='font-size:32px;font-weight:700;margin-top:14px'>{t}</div></div>" for e, t in [("📱","Reconoce tu cara"),("🎧","Recomienda música"),("🗺️","Encuentra rutas"),("💬","Entiende tu voz")])}
         </div>{foot()}</div>"""),
        # 2. Nombre
        (f"Te presentamos {PROGRAMA}, de la Escuela de Ciencias Básicas, Tecnología e Ingeniería de la UNAD. "
         "Es un programa profesional totalmente virtual, de ciento cincuenta créditos, que te forma como ingeniero o ingeniera.",
         f"""<div class='s grid-bg'>
         <span class='tag'>Programa profesional · ECBTI</span>
         <h1 style='font-size:104px;margin-top:50px'>Ingeniería en<br><span class='gold'>Inteligencia Artificial</span><br>y <span class='teal'>Ciencia de Datos</span></h1>
         <div style='display:flex;gap:30px;margin-top:70px'>
           {''.join(f"<div style='background:rgba(255,255,255,.1);border-radius:24px;padding:26px 40px'><div style='font-size:56px;font-weight:900;color:var(--gold)'>{a}</div><div style='font-size:28px;font-weight:600'>{b}</div></div>" for a, b in [("100 %","Virtual"),("150","Créditos"),("Título","Ingeniero(a)")])}
         </div>{foot()}</div>"""),
        # 3. Qué hace
        ("¿Y qué hace este profesional? Enseña a las máquinas a aprender. Diseña sistemas inteligentes que automatizan tareas, "
         "analiza grandes cantidades de datos para tomar mejores decisiones y crea soluciones con visión por computador, "
         "lenguaje natural e inteligencia artificial generativa. Y lo hace siempre con ética, seguridad y compromiso social.",
         f"""<div class='s'>
         <span class='tag'>¿Qué hace el profesional?</span>
         <h2>Enseña a las máquinas <span class='gold'>a aprender</span></h2>
         <div class='cards'>{que_html}</div>{foot()}</div>"""),
        # 4. Dónde trabaja
        ("¿Dónde puedes trabajar? Como científico o analista de datos, desarrollador de soluciones de inteligencia artificial, "
         "especialista en visualización de datos, líder de proyectos o consultor en transformación digital. "
         "Tu trabajo se necesita en salud, finanzas, comercio, logística, educación, el campo y el sector público. "
         "¡Y también puedes crear tu propia empresa!",
         f"""<div class='s' style='padding-top:90px'>
         <span class='tag'>¿En qué puedes trabajar?</span>
         <div style='display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:44px;width:100%'>{cargos_html}</div>
         <div style='font-size:34px;font-weight:800;margin:46px 0 20px'>En sectores como:</div>
         <div style='display:flex;flex-wrap:wrap;gap:16px'>{sectores_html}</div>{foot()}</div>"""),
        # 5. Articulación
        ("Y lo mejor es que puedes empezar desde ya. Mientras cursas los ciclos cinco y seis del SINEP, puedes matricular cursos "
         "articulados como " + ", ".join(n for n, _, _ in CURSOS_ARTICULACION[:-1]) + " e " + CURSOS_ARTICULACION[-1][0] +
         ". Cuando los apruebes, esos créditos se te reconocen al entrar al programa. Así ahorras tiempo y dinero.",
         f"""<div class='s'>
         <span class='tag'>Cursos de articulación SINEP</span>
         <h2>Empieza la universidad <span class='gold'>desde el colegio</span></h2>
         <div class='cards' style='margin-top:56px'>{cursos_html}</div>
         <div style='margin-top:44px;font-size:36px;font-weight:700'>✔ Hasta <span class='gold'>{total} créditos</span> reconocidos &nbsp; ✔ Ahorras tiempo &nbsp; ✔ Ahorras dinero</div>
         {foot()}</div>"""),
        # 6. Cierre
        ("Tu ruta hacia la educación superior comienza aquí. Del SINEP a la Ingeniería. "
         "Llena el formulario de interés y acompáñanos en la jornada de Tecnologías Digitales. ¡Te esperamos en la UNAD!",
         f"""<div class='s grid-bg' style='display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center'>
         <div class='circle' style='width:1300px;height:1300px;left:310px;top:-110px'></div>
         <span class='tag'>Tu ruta hacia la educación superior comienza aquí</span>
         <h1 style='font-size:128px;margin-top:40px'>Del SINEP a la<br><span class='gold'>Ingeniería</span></h1>
         <div style='margin-top:50px;font-size:38px;font-weight:700'>📝 Llena el formulario de interés</div>
         <div style='margin-top:26px;background:var(--gold);color:var(--navy);font-size:34px;font-weight:800;padding:18px 40px;border-radius:50px'>{JORNADA}</div>
         <div class='bar'></div></div>"""),
    ]


def dur(wav):
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voz", required=True)
    ap.add_argument("--tmp", default="/tmp/video_iacd")
    a = ap.parse_args()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    out = OUT / "video"; out.mkdir(exist_ok=True)

    sc = escenas()
    render([(doc(html, SCENE_CSS), tmp / f"e{i}.png") for i, (_, html) in enumerate(sc)], W, H)

    clips, srt, t0 = [], [], 0.0
    for i, (texto, _) in enumerate(sc):
        wav = tmp / f"e{i}.wav"
        subprocess.run(["python3", "-m", "piper", "-m", a.voz, "-f", str(wav), "--sentence_silence", "0.2"],
                       input=texto.replace("UNAD", "Unad").replace("SINEP", "Sinep").encode(), check=True, capture_output=True)
        d = dur(wav) + 0.9  # 0.5 s antes + 0.4 s después
        clip = tmp / f"e{i}.mp4"
        subprocess.run([FFMPEG, "-y", "-loop", "1", "-i", str(tmp / f"e{i}.png"), "-i", str(wav),
                        "-filter_complex",
                        f"[0:v]scale={W}:{H},format=yuv420p,fade=t=in:st=0:d=0.35,fade=t=out:st={d-0.35:.2f}:d=0.35[v];"
                        f"[1:a]adelay=500|500,apad,atrim=0:{d:.2f},aformat=sample_rates=44100:channel_layouts=stereo[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{d:.2f}", "-r", "30", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k", str(clip)],
                       check=True, capture_output=True)
        clips.append(clip)
        srt.append(f"{i+1}\n{srt_time(t0 + 0.5)} --> {srt_time(t0 + d - 0.4)}\n{texto}\n")
        t0 += d

    lst = tmp / "lista.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    final = out / "video_IA_CD_SINEP.mp4"
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True, capture_output=True)
    (out / "subtitulos.srt").write_text("\n".join(srt), encoding="utf-8")
    for i in (0, 4):
        (out / f"fotograma_{i+1}.png").write_bytes((tmp / f"e{i}.png").read_bytes())
    print(f"Duración total: {t0:.1f} s -> {final}")


if __name__ == "__main__":
    main()
