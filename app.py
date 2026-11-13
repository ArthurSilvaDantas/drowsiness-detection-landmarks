"""Aplicação em vídeo: landmarks, abertura dos olhos (EAR), análise temporal e alerta.

    python app.py --source 0                                      # webcam
    python app.py --source video.mp4 --out saida.mp4 --no-display

Fluxo por frame: YuNet acha a maior face -> recorte igual ao do treino -> 98 landmarks ->
EAR (média dos dois olhos) -> média móvel -> máquina de estados com histerese -> alerta.

Calibração: nos primeiros segundos com face, a pessoa fica de olhos abertos; o EAR típico é a
mediana desse trecho (robusta a piscadas) e os limiares são frações dele:
    fecha quando EAR < close_ratio x mediana; reabre quando EAR > open_ratio x mediana.

Cada fechamento é classificado ao reabrir pela sua duração:
    piscada            <= blink_max segundos
    fechamento curto   <  alert_seconds
    fechamento longo   >= alert_seconds  (o alerta acende assim que a duração passa do limite)

Tempo: na webcam, relógio do sistema; em arquivo, índice do frame / FPS do arquivo.
Sem face no frame: mantém o último estado; se os olhos estavam fechados, o tempo continua contando
(uma cabeça que cai também é sinal de sonolência). Quedas de 1 frame são absorvidas pela média móvel.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from predict import LandmarkModel
from utils import EAR6_LEFT, EAR6_RIGHT

ROOT = Path(__file__).resolve().parent
EAR8_RIGHT = (60, 64, [(61, 67), (62, 66), (63, 65)])
EAR8_LEFT = (72, 68, [(71, 73), (70, 74), (69, 75)])
RED, GREEN, YELLOW, WHITE, GRAY = (0, 0, 255), (0, 200, 0), (0, 220, 255), (255, 255, 255), (160, 160, 160)


# ------------------------------------------------------------------------------- medida dos olhos
def eye_ear6(p, idx):
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9)


def eye_ear8(p, spec):
    a, b, pairs = spec
    return sum(np.linalg.norm(p[u] - p[v]) for u, v in pairs) / (3 * np.linalg.norm(p[a] - p[b]) + 1e-9)


def ear(points, kind="ear6"):
    """EAR médio dos dois olhos (pontos do WFLW)."""
    if kind == "ear8":
        return (eye_ear8(points, EAR8_RIGHT) + eye_ear8(points, EAR8_LEFT)) / 2
    return (eye_ear6(points, EAR6_RIGHT) + eye_ear6(points, EAR6_LEFT)) / 2


# ------------------------------------------------------------------------------- análise temporal
class EyeMonitor:
    """Calibração, suavização, histerese e classificação dos fechamentos."""

    def __init__(self, calib_seconds=3.0, close_ratio=0.7, open_ratio=0.8, smooth=3,
                 blink_max=0.4, alert_seconds=1.5, fixed_threshold=None):
        self.calib_seconds, self.close_ratio, self.open_ratio = calib_seconds, close_ratio, open_ratio
        self.blink_max, self.alert_seconds = blink_max, alert_seconds
        self.window = deque(maxlen=max(1, smooth))
        self.calib, self.calib_time, self.last_t = [], 0.0, None
        self.baseline = None
        if fixed_threshold is not None:   # sem calibração: limiar fixo informado pelo usuário
            self.baseline = fixed_threshold / close_ratio
        self.closed, self.closed_since = False, None
        self.alert = False
        self.events = {"piscada": 0, "fechamento curto": 0, "fechamento longo": 0}
        self.last_event = None

    @property
    def t_close(self):
        return None if self.baseline is None else self.close_ratio * self.baseline

    @property
    def t_open(self):
        return None if self.baseline is None else self.open_ratio * self.baseline

    def update(self, t, value):
        """t em segundos; value = EAR do frame. Retorna o EAR suavizado."""
        self.window.append(value)
        smoothed = float(np.mean(self.window))
        if self.baseline is None:                          # ainda calibrando
            # conta só o tempo com face (intervalos sem face maiores que 0,5 s não entram)
            if self.last_t is not None and t - self.last_t < 0.5:
                self.calib_time += t - self.last_t
            self.last_t = t
            self.calib.append(value)
            if self.calib_time >= self.calib_seconds and len(self.calib) >= 10:
                self.baseline = float(np.median(self.calib))
            return smoothed
        if not self.closed and smoothed < self.t_close:
            self.closed, self.closed_since = True, t
        elif self.closed and smoothed > self.t_open:
            self._finish(t)
        self.alert = self.closed and (t - self.closed_since) >= self.alert_seconds
        return smoothed

    def _finish(self, t):
        d = t - self.closed_since
        kind = ("piscada" if d <= self.blink_max else
                "fechamento curto" if d < self.alert_seconds else "fechamento longo")
        self.events[kind] += 1
        self.last_event = (kind, d, t)
        self.closed, self.closed_since, self.alert = False, None, False

    def closed_for(self, t):
        return 0.0 if not self.closed else t - self.closed_since


# ------------------------------------------------------------------------------- desenho
def draw(frame, pts, box, mon: EyeMonitor, ear_s, t, history, fps_proc, no_face):
    h, w = frame.shape[:2]
    if pts is not None:
        r = max(3, round(max(h, w) / 450))          # pontos um pouco maiores em vídeos de alta resolução
        for k, (x, y) in enumerate(pts.astype(int)):
            color = YELLOW if 60 <= k <= 75 else GREEN
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), GRAY, 1)

    # painel de texto
    lines = []
    if mon.baseline is None:
        lines.append(("CALIBRANDO: mantenha os olhos abertos", YELLOW))
    else:
        state = "FECHADO" if mon.closed else "ABERTO"
        lines.append((f"olhos: {state}", RED if mon.closed else GREEN))
    lines.append((f"EAR: {ear_s:.3f}" if ear_s is not None else "EAR: -", WHITE))
    if mon.baseline is not None:
        lines.append((f"limiar: fecha < {mon.t_close:.3f} | abre > {mon.t_open:.3f}", WHITE))
    if mon.closed:
        lines.append((f"fechado ha {mon.closed_for(t):.1f} s", RED))
    e = mon.events
    lines.append((f"piscadas {e['piscada']} | curtos {e['fechamento curto']} | longos {e['fechamento longo']}", WHITE))
    if mon.last_event:
        kind, d, te = mon.last_event
        if t - te < 2.0:
            lines.append((f"ultimo: {kind} ({d:.2f} s)", YELLOW))
    if no_face:
        lines.append(("FACE NAO DETECTADA (mantendo o ultimo estado)", YELLOW))
    lines.append((f"{fps_proc:.0f} fps | t = {t:.1f} s", GRAY))

    panel_h = 22 * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(w, 520), panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # gráfico do EAR nos últimos segundos
    gh, gw = 80, min(w - 20, 400)
    gx, gy = w - gw - 10, h - gh - 10
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
    if len(history) > 1:
        t_end, span = history[-1][0], 10.0
        to_xy = lambda tt, v: (int(gx + gw * (1 - (t_end - tt) / span)), int(gy + gh * (1 - min(v, 0.5) / 0.5)))
        pts_line = [to_xy(tt, v) for tt, v in history if t_end - tt <= span]
        cv2.polylines(frame, [np.array(pts_line)], False, GREEN, 1, cv2.LINE_AA)
        if mon.t_close is not None:
            yy = to_xy(t_end, mon.t_close)[1]
            cv2.line(frame, (gx, yy), (gx + gw, yy), RED, 1)
    cv2.putText(frame, "EAR (10 s)", (gx + 5, gy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    if mon.alert:   # alerta bem visível: borda vermelha + texto
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 14)
        text = "ALERTA: OLHOS FECHADOS"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
        cv2.putText(frame, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.2, RED, 3, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------------------- principal
def open_writer(path, shape, fps):
    h, w = shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        sys.exit(f"não consegui criar o vídeo de saída: {path}")
    return writer


def default_weights():
    for p in (ROOT / "pesos" / "modelo.pt", ROOT / "weights" / "modelo.pt"):
        if p.is_file():
            return str(p)
    return str(ROOT / "pesos" / "modelo.pt")


def main():
    p = argparse.ArgumentParser(description="Detecção de sonolência em vídeo com o modelo de landmarks.")
    p.add_argument("--source", default="0", help="índice da webcam (ex.: 0) ou caminho do vídeo")
    p.add_argument("--out", default=None, help="salva o vídeo processado (ex.: saida.mp4)")
    p.add_argument("--no-display", action="store_true", help="processa sem abrir janela")
    p.add_argument("--weights", default=default_weights(), help="pesos do modelo (padrão: pesos/modelo.pt)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ear", choices=["ear6", "ear8"], default="ear6", help="pontos usados no EAR")
    p.add_argument("--calib-seconds", type=float, default=3.0, help="duração da calibração com olhos abertos")
    p.add_argument("--close-ratio", type=float, default=0.7, help="fecha quando EAR < fração x mediana")
    p.add_argument("--open-ratio", type=float, default=0.8, help="reabre quando EAR > fração x mediana")
    p.add_argument("--threshold", type=float, default=None, help="limiar fixo (desliga a calibração)")
    p.add_argument("--smooth", type=int, default=3, help="frames da média móvel do EAR")
    p.add_argument("--blink-max", type=float, default=0.4, help="duração máxima de uma piscada (s)")
    p.add_argument("--alert-seconds", type=float, default=1.5, help="fechamento que gera alerta (s)")
    p.add_argument("--no-tta", action="store_true", help="desliga a média com a face espelhada (mais rápido)")
    p.add_argument("--log", default=None, help="salva um CSV com tempo, EAR e estado por frame")
    args = p.parse_args()

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        sys.exit(f"não consegui abrir a fonte de vídeo: {args.source}")
    fps_file = cap.get(cv2.CAP_PROP_FPS)
    if not 1.0 <= (fps_file or 0) <= 120.0:   # alguns arquivos e câmeras devolvem 0 ou valores absurdos
        fps_file = 30.0
    if args.open_ratio <= args.close_ratio:
        sys.exit("--open-ratio precisa ser maior que --close-ratio (histerese)")

    model = LandmarkModel(args.weights, device=args.device)
    model.detector.max_side = 640          # detecção mais rápida
    model.flip_tta = not args.no_tta
    mon = EyeMonitor(args.calib_seconds, args.close_ratio, args.open_ratio, args.smooth,
                     args.blink_max, args.alert_seconds, args.threshold)

    writer, log_rows, history = None, [], deque(maxlen=600)
    pending = []   # webcam: guarda os primeiros frames para medir o fps real antes de abrir o vídeo
    last_pts = last_box = None
    ear_s, frame_idx, t0 = None, 0, time.monotonic()
    tick = time.monotonic()
    fps_proc = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic() - t0 if is_cam else frame_idx / fps_file
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = model.detector.detect(frame)
        no_face = len(boxes) == 0
        if not no_face:
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]   # maior face
            pts = model.predict_boxes(rgb, box[None])[0]
            ear_s = mon.update(t, ear(pts, args.ear))
            history.append((t, ear_s))
            last_pts, last_box = pts, box
        elif mon.closed:                    # sem face: mantém o último estado e o relógio
            mon.alert = (t - mon.closed_since) >= mon.alert_seconds

        now = time.monotonic()
        fps_proc = 0.9 * fps_proc + 0.1 / max(now - tick, 1e-6) if fps_proc else 1 / max(now - tick, 1e-6)
        tick = now
        out = draw(frame, None if no_face else last_pts, last_box, mon, ear_s, t, history, fps_proc, no_face)

        if args.out:
            if writer is None:
                pending.append((t, out))
                # arquivo: usa o fps do arquivo; webcam: mede o fps real nos primeiros ~2 s, para o
                # vídeo gravado ter a mesma duração da cena (senão piscadas e alertas mudam de duração)
                if not is_cam or len(pending) >= 40 or t - pending[0][0] >= 2.0:
                    fps_out = fps_file if not is_cam else (len(pending) - 1) / max(t - pending[0][0], 1e-3)
                    writer = open_writer(args.out, out.shape, fps_out)
                    for _, f in pending:
                        writer.write(f)
                    pending = []
            else:
                writer.write(out)
        if args.log:
            log_rows.append({"t": round(t, 4), "ear": None if no_face else round(ear_s, 4),
                             "closed": int(mon.closed), "alert": int(mon.alert), "face": int(not no_face),
                             "t_close": mon.t_close})
        if not args.no_display:
            cv2.imshow("deteccao de sonolencia (q para sair)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if mon.closed:                           # fecha o último evento no fim do vídeo
        mon._finish(t)
    if args.out and writer is None and pending:  # vídeo curto demais para medir o fps
        writer = open_writer(args.out, pending[0][1].shape, fps_file)
        for _, f in pending:
            writer.write(f)
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if args.log and log_rows:
        with open(args.log, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            wr.writeheader()
            wr.writerows(log_rows)
    print(f"frames: {frame_idx} | EAR típico: {mon.baseline if mon.baseline is None else round(mon.baseline, 3)} | "
          f"eventos: {mon.events}")


if __name__ == "__main__":
    main()
