"""Gera apresentacoes/apresentacao_rf_do_zero.html a partir das saídas dos notebooks passo_01..07b.

Todos os números dos slides vêm de saidas/*.csv, saidas/*.pkl ou são recalculados aqui com o
mesmo código dos notebooks (rf_zero.py). Os poucos valores que só existem na saída impressa de
um notebook estão marcados com # fonte: <notebook>.

Formato dos slides de teste: à esquerda "Como foi feito" (passos e cálculo) e "Decisão"
(o que escolhemos e o que pesou); à direita os resultados em gráficos.

Rodar de dentro de frentes/rf_do_zero:  python gerar_slides.py
"""
import html
import numpy as np
import pandas as pd
import rf_zero as rz

SAIDA = "../../apresentacoes/apresentacao_rf_do_zero.html"
AZUL, LARANJA, VERDE, VERMELHO, CINZA, TINTA, TINTA2, GRADE = (
    "#1F5FA8", "#C0661A", "#3F8F5A", "#B3423A", "#9AA6A2", "#12233D", "#4A5A6E", "#C9D3D0")
AZUL_MEDIO = "#7FA6D3"
W = 680  # largura (viewBox) dos gráficos da coluna da direita


# ------------------------------------------------------------------ formatação pt-BR
def f(x, d=1, sinal=False):
    s = f"{x:+,.{d}f}" if sinal else f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def rs(x):
    return ("−" if x < 0 else "") + "R$ " + f(abs(x), 0)


def esc(s):
    return html.escape(str(s))


# ------------------------------------------------------------------ dados
print("carregando dados...")
win_1m = rz.carregar("../../dados/WINN_M1/WIN$N_M1.csv")

# Teste 1 (passo 1): leitura dos dados por candle, só no período de treino
treino_1m = win_1m[win_1m.index < rz.VALIDACAO[0]]
viab = {}
for m in [1, 5, 15, 30, 60]:
    b = rz.adicionar_alvo(rz.reamostrar(treino_1m, m)).dropna(subset=["mov_prox"])
    a = b["mov_prox"].abs()
    viab[m] = dict(n=len(b), mov=a.mean(), pmin=100 * (1 + rz.CUSTO_PTS / a.mean()) / 2,
                   por_dia=len(b) / b.index.normalize().nunique(),
                   sempre=rz.resultado_operacoes(np.ones(len(b)), b["mov_prox"].values).mean())
n_pregoes_treino = treino_1m.index.normalize().nunique()

# Teste 2 (passo 2): movimento por ano (candle 15 min) e autocorrelação
b15 = rz.adicionar_alvo(rz.reamostrar(win_1m, 15)).dropna(subset=["ret_prox"])
por_ano = pd.DataFrame({"pts": b15["mov_prox"].abs().groupby(b15.index.year).mean(),
                        "pct": b15["ret_prox"].abs().groupby(b15.index.year).mean(),
                        "preco": b15["close"].groupby(b15.index.year).mean()})
r15_tr = b15[b15.index < rz.VALIDACAO[0]]["ret_prox"]
curtose15 = r15_tr.kurt()
autocorr15 = [r15_tr.autocorr(k) for k in range(1, 6)]
faixa15 = 2 / np.sqrt(len(r15_tr))

# Teste 3 (passo 3)  # fonte: passo_03_variaveis.ipynb (tabelas tab_lags, tab_grupos e tab_imp)
lags_bruto = pd.DataFrame({"3 lags": [0.859, -4.423, -2.787, -2.767], "5 lags": [1.022, -2.882, 3.496, 7.138],
                           "10 lags": [0.628, -2.375, 0.855, 1.634],
                           "5 lags + todos os grupos": [0.629, -0.424, 5.837, 12.927]},
                          index=["5 min", "15 min", "30 min", "60 min"])
imp = pd.DataFrame({"15 min": [-0.0668, -0.0181, 0.0137, 0.0540, 0.0241],
                    "60 min": [0.2349, 0.0818, 0.2114, 1.1807, 0.1252]},
                   index=["lags", "forma", "volatilidade", "volume", "contexto do dia"])

# Teste 4 (passo 4.1)  # fonte: passo_04_candle_janela_retreino.ipynb, tabela "curva"
env_acerto = {"15 min": [50.26, 49.44, 50.26, 49.49, 50.28, 49.50],
              "60 min": [49.20, 50.46, 51.76, 51.48, 52.25, 54.11]}
g4 = pd.read_csv("saidas/passo04_grade.csv", index_col=[0, 1, 2])

# Teste 6 (passo 5): regras
r5 = pd.read_csv("saidas/passo05_regras.csv", index_col=[0, 1])
d60 = pd.read_pickle("saidas/passo04_previsoes.pkl")[(60, 36, "trimestral")]
pts60 = rz.prev_pontos(d60)
curvas5 = {}
for rot, k in [("opera sempre", 0), ("previsto > custo", 1), ("previsto > 2× custo", 2), ("previsto > 3× custo", 3)]:
    dirc = np.sign(pts60) if k == 0 else np.sign(pts60).where(pts60.abs() > k * rz.CUSTO_PTS, 0)
    op = dirc != 0
    res = pd.Series(rz.resultado_operacoes(dirc.values, d60["mov_prox"].values), index=d60.index[op.values])
    curvas5[rot] = res.cumsum() * rz.REAIS_POR_PONTO["WIN"]
assert abs(curvas5["previsto > custo"].iloc[-1] - r5.loc[("60 min × 36 m", "prev > 1×custo"), "total R$ (1 contrato)"]) < 1

# Teste 7 (passo 6): hiperparâmetros
g6 = pd.read_csv("saidas/passo06_grade.csv", index_col=[0, 1, 2])
ESC = (4, 0.01, 0.5)

# Resultado (passos 7 e 7b)
p7 = pd.read_pickle("saidas/passo07_previsoes.pkl")
p7b = pd.read_pickle("saidas/passo07b_previsoes.pkl")
r7 = pd.read_csv("saidas/passo07b_resultado.csv", index_col=0)


def ops_regra(d):
    pts = rz.prev_pontos(d)
    dirc = np.sign(pts).where(pts.abs() > rz.CUSTO_PTS, 0)
    op = dirc != 0
    return pd.Series(rz.resultado_operacoes(dirc.values, d["mov_prox"].values), index=d.index[op.values]), \
        d["mov_prox"].values[op.values]


res_teste, mov_teste = ops_regra(p7["teste 01/25–03/26"])
res_extra, mov_extra = ops_regra(p7b)

# Buracos na base: meses com menos da metade dos candles típicos
cont_mes = win_1m["close"].resample("MS").count()
tipico = cont_mes.median()


# ------------------------------------------------------------------ helpers SVG
def svg(w, h, corpo, defs=""):
    return f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg">{defs}{corpo}</svg>'


def txt(x, y, s, size=12, cor=TINTA2, anchor="start", peso=400):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{cor}" text-anchor="{anchor}" font-weight="{peso}">{esc(s)}</text>'


def linha(x1, y1, x2, y2, cor=GRADE, w=1, extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{cor}" stroke-width="{w}" {extra}/>'


def ret(x, y, w, h, cor, extra=""):
    if h < 0:
        y, h = y + h, -h
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{cor}" {extra}/>'


def poli(pts, cor, w=2.4):
    return f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{cor}" stroke-width="{w}"/>'


def escala(vmin, vmax, top, bot):
    return lambda v: bot - (v - vmin) / (vmax - vmin) * (bot - top)


def eixo_y(x0, x1, y_de, ticks, fmt=lambda v: f(v, 0)):
    out = ""
    for v in ticks:
        y = y_de(v)
        out += linha(x0, y, x1, y, TINTA2 if v == 0 else GRADE, 1.3 if v == 0 else 1)
        out += txt(x0 - 6, y + 4, fmt(v), 11, TINTA2, "end")
    return out


def titulo(s, x=4, y=15):
    return txt(x, y, s, 13, TINTA, "start", 700)


def heat_cor(v, lim):
    t = max(-1.0, min(1.0, v / lim))
    a = np.array([248, 249, 247])
    b = np.array([63, 143, 90]) if t >= 0 else np.array([179, 66, 58])
    c = (a + (b - a) * abs(t)).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})", abs(t) > 0.55


def barras(x0, x1, top, bot, rotulos, valores, cor, vmax, ticks, fmt_eixo, fmt_val, sub=None, vmin=0):
    y_de = escala(vmin, vmax, top, bot)
    out = eixo_y(x0, x1, y_de, ticks, fmt_eixo)
    bw = (x1 - x0) / len(rotulos)
    for i, (r, v) in enumerate(zip(rotulos, valores)):
        x = x0 + i * bw + bw * 0.2
        c = cor(v) if callable(cor) else cor
        out += ret(x, y_de(0), bw * 0.6, y_de(v) - y_de(0), c)
        out += txt(x + bw * 0.3, (y_de(v) - 6) if v >= 0 else (y_de(v) + 15), fmt_val(v), 11.5, TINTA, "middle", 700)
        out += txt(x + bw * 0.3, bot + 16, r, 11.5, TINTA2, "middle")
        if sub:
            out += txt(x + bw * 0.3, bot + 30, sub[i], 10.5, CINZA, "middle")
    return out, y_de


# ------------------------------------------------------------------ gráficos: abertura
def g_fluxo():
    caixas = [("Candles do WIN", "1 min → 60 min"), ("17 variáveis", "o que se sabe em t"),
              ("Random Forest", "treinada sem custo"), ("Movimento previsto", "em pontos, t → t+1"),
              ("Regra", "opera se paga o custo"), ("Resultado", "pts e R$, com custo")]
    bw, gap = 166, 32
    out = ""
    for i, (a, b) in enumerate(caixas):
        x = 4 + i * (bw + gap)
        destaque = i in (2, 3)
        out += ret(x, 20, bw, 76, AZUL if destaque else "#FFFFFF", f'stroke="{AZUL if destaque else GRADE}" rx="4"')
        out += txt(x + bw / 2, 52, a, 15, "#fff" if destaque else TINTA, "middle", 700)
        out += txt(x + bw / 2, 74, b, 12, "#E8EEF7" if destaque else TINTA2, "middle")
        if i < len(caixas) - 1:
            out += linha(x + bw + 4, 58, x + bw + gap - 6, 58, TINTA2, 1.5)
            out += f'<path d="M{x + bw + gap - 6},58 l-7,-4 v8 z" fill="{TINTA2}"/>'
    out += ret(4 + 4 * (bw + gap) - 6, 104, 2 * bw + gap + 12, 3, LARANJA)
    out += txt(4 + 4 * (bw + gap) + bw + gap / 2, 120, "o custo de 5 pts entra aqui, não no treino", 12, LARANJA, "middle", 650)
    return svg(1160, 126, out)


def g_linha_tempo():
    w, h = 1160, 250
    x0, x1 = 120, 1150
    ini, fim = pd.Timestamp("2020-10-01"), pd.Timestamp("2026-10-01")
    X = lambda t: x0 + (pd.Timestamp(t) - ini).days / (fim - ini).days * (x1 - x0)
    out, y = "", 34
    out += txt(10, y + 18, "Divisão", 13, TINTA, "start", 700)
    for a, b, cor, op in [(ini, rz.VALIDACAO[0], AZUL, 0.55), (rz.VALIDACAO[0], rz.VALIDACAO[1], AZUL, 1),
                          (rz.TESTE[0], rz.TESTE[1], LARANJA, 1), (rz.EXTRA[0], "2026-09-26", VERDE, 1)]:
        out += ret(X(a), y, X(b) - X(a) - 2, 26, cor, f'fill-opacity="{op}"')
    for mth, c in cont_mes.items():
        if mth >= pd.Timestamp("2020-11-01") and c < 0.5 * tipico:
            out += ret(X(mth), y, X(mth + pd.DateOffset(months=1)) - X(mth), 26, "url(#hach)")
    for nome, a, b, cor in [("treino · 15/10/2020 → 12/2023", ini, rz.VALIDACAO[0], AZUL),
                            ("validação · 2024", rz.VALIDACAO[0], rz.VALIDACAO[1], AZUL),
                            ("teste · 01/25 → 03/26", rz.TESTE[0], rz.TESTE[1], LARANJA), ("extra", rz.EXTRA[0], rz.EXTRA[1], VERDE)]:
        out += txt((X(a) + X(b)) / 2, y - 8, nome, 12, cor, "middle", 700)
    out += txt(X("2023-08-15"), y + 44, "buracos na base (o maior: jun–out/2023)", 11, TINTA2, "middle")
    for ano in range(2021, 2027):
        xa = X(f"{ano}-01-01")
        out += linha(xa, y + 30, xa, h - 18, GRADE, 1, 'stroke-dasharray="2 3"')
        out += txt(xa, h - 4, str(ano), 11, TINTA2, "middle")
    out += txt(10, 112, "Re-treino", 13, TINTA, "start", 700)
    for k, s in enumerate(["trimestral,", "janela de", "36 meses"]):
        out += txt(10, 128 + 14 * k, s, 11, TINTA2)
    blocos = [("2024-01-01", "2024-04-01"), ("2024-04-01", "2024-07-01"), ("2024-07-01", "2024-10-01"),
              ("2024-10-01", "2025-01-01"), ("2025-01-01", "2025-04-01")]
    for i, (a, b) in enumerate(blocos):
        yy, ta = 96 + i * 24, pd.Timestamp(a)
        out += ret(X(ta - pd.DateOffset(months=36)), yy, X(ta) - X(ta - pd.DateOffset(months=36)) - 2, 16, AZUL, 'fill-opacity="0.35"')
        out += ret(X(ta), yy, X(b) - X(ta), 16, LARANJA)
        if i == 0:
            out += txt(X(ta - pd.DateOffset(months=18)), yy + 12, "aprende com os últimos 36 meses", 11, TINTA, "middle", 600)
            out += txt(X(b) + 6, yy + 12, "prevê os 3 meses seguintes", 11, LARANJA, "start", 650)
    out += txt(X("2025-04-01") + 6, 96 + 4 * 24 + 12, "… e assim por diante", 11, TINTA2)
    defs = (f'<defs><pattern id="hach" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="#fff"/><line x1="0" y1="0" x2="0" y2="6" stroke="{CINZA}" stroke-width="3"/></pattern></defs>')
    return svg(w, h, out, defs)


def g_acerto_exemplo():
    y_de = escala(-12, 6, 30, 170)
    out = titulo("Exemplo: acertar 55% e ainda perder dinheiro", 60, 18) + linha(60, y_de(0), 500, y_de(0), TINTA2, 1.2)
    for i, (a, b, v, cor) in enumerate([("55% das vezes", "ganha 2 pts", 2, VERDE), ("45% das vezes", "perde 10 pts", -10, VERMELHO),
                                        ("custo", "5 pts", -5, LARANJA), ("esperança", "por operação", 0.55 * 2 - 0.45 * 10 - 5, TINTA)]):
        x = 80 + i * 108
        out += ret(x, y_de(0), 70, y_de(v) - y_de(0), cor, 'fill-opacity="0.85"')
        out += txt(x + 35, (y_de(v) - 6) if v > 0 else (y_de(v) + 15), f(v, 1, True), 13, cor, "middle", 700)
        out += txt(x + 35, 190, a, 11.5, TINTA, "middle", 650) + txt(x + 35, 204, b, 11, TINTA2, "middle")
    return svg(520, 210, out)


# ------------------------------------------------------------------ gráficos: testes (coluna da direita)
def t1_movimento():
    ms = [1, 5, 15, 30, 60]
    out = titulo("Movimento médio do próximo candle (pts) · treino 2020–2023")
    o, y_de = barras(50, W - 10, 34, 190, [f"{m} min" for m in ms], [viab[m]["mov"] for m in ms], AZUL, 350,
                     [0, 100, 200, 300], lambda v: f(v, 0), lambda v: f(v, 0),
                     sub=[f"{viab[m]['por_dia']:.0f} candles/dia" for m in ms])
    out += o + linha(50, y_de(5), W - 10, y_de(5), LARANJA, 2, 'stroke-dasharray="5 3"')
    out += linha(60, 40, 84, 40, LARANJA, 2, 'stroke-dasharray="5 3"') + txt(90, 44, "custo de 5 pts (régua) – linha no pé das barras", 11, LARANJA, "start", 700)
    return svg(W, 228, out)


def t1_acerto():
    ms = [1, 5, 15, 30, 60]
    x0, x1, top, bot = 50, W - 10, 34, 170
    y_de = escala(48, 58, top, bot)
    out = titulo("Acerto de direção necessário só para empatar: p = (1 + 5 ÷ M) ÷ 2")
    out += eixo_y(x0, x1, y_de, [48, 50, 52, 54, 56, 58], lambda v: f"{v:.0f}%")
    out += linha(x0, y_de(50), x1, y_de(50), TINTA2, 1, 'stroke-dasharray="2 3"')
    bw = (x1 - x0) / len(ms)
    pts = [(x0 + i * bw + bw / 2, y_de(viab[m]["pmin"])) for i, m in enumerate(ms)]
    out += poli(pts, LARANJA, 2.5)
    for (x, y), m in zip(pts, ms):
        out += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{LARANJA}"/>'
        out += txt(x, y - 10, f(viab[m]["pmin"], 1) + "%", 12, TINTA, "middle", 700)
        out += txt(x, bot + 16, f"{m} min", 11.5, TINTA2, "middle")
    out += txt(x0 + 4, y_de(50) + 13, "50% = moeda", 10.5, TINTA2, "start")
    return svg(W, 192, out)


def t2_anos():
    anos = list(por_ano.index)
    out = titulo("Movimento médio do candle de 15 min, por ano")
    o, _ = barras(50, W / 2 - 14, 40, 186, [str(a) for a in anos], por_ano["pts"].values, LARANJA, 240,
                  [0, 80, 160, 240], lambda v: f(v, 0), lambda v: f(v, 0),
                  sub=[f"{p / 1000:.0f} mil" for p in por_ano["preco"]])
    out += o + txt(50, 34, "em pontos", 11.5, LARANJA, "start", 700)
    o, _ = barras(W / 2 + 40, W - 10, 40, 186, [str(a) for a in anos], por_ano["pct"].values, AZUL, 0.16,
                  [0, 0.05, 0.10, 0.15], lambda v: f(v, 2), lambda v: f(v, 2))
    out += o + txt(W / 2 + 40, 34, "em % do preço", 11.5, AZUL, "start", 700)
    out += txt(50, 228, "embaixo dos anos: preço médio do WIN", 10.5, CINZA)
    return svg(W, 234, out)


def t2_autocorr():
    x0, x1, top, bot = 50, W - 10, 34, 160
    y_de = escala(-0.03, 0.03, top, bot)
    out = titulo("Autocorrelação do retorno (15 min, treino): t com t+1 … t+5")
    out += eixo_y(x0, x1, y_de, [-0.03, -0.015, 0, 0.015, 0.03], lambda v: f(v, 3))
    out += ret(x0, y_de(faixa15), x1 - x0, y_de(-faixa15) - y_de(faixa15), CINZA, 'fill-opacity="0.25"')
    out += txt(x1 - 4, y_de(faixa15) - 4, f"faixa de ruído ±2/√n = ±{f(faixa15, 3)}", 10.5, TINTA2, "end")
    bw = (x1 - x0) / 5
    for i, v in enumerate(autocorr15):
        x = x0 + i * bw + bw * 0.3
        out += ret(x, y_de(0), bw * 0.4, y_de(v) - y_de(0), AZUL)
        out += txt(x + bw * 0.2, bot + 16, f"defasagem {i + 1}", 11, TINTA2, "middle")
    return svg(W, 182, out)


def t3_tabela_lags():
    linhas = ""
    for c in lags_bruto.index:
        vals = lags_bruto.loc[c]
        melhor = vals.idxmax()
        linhas += f"<tr><td><strong>{c}</strong></td>" + "".join(
            f"<td class='n'{' style=\"font-weight:700;color:#12233D\"' if k == melhor else ''}>{f(v, 1, True)}</td>"
            for k, v in vals.items()) + "</tr>"
    return (f'<div class="rotulo-res">Ganho bruto por operação (pts, sem custo) · validação 2024 · em negrito o melhor de cada candle</div>'
            f'<table class="tab"><tr><th>Candle</th><th class="n">3 lags</th><th class="n">5 lags</th><th class="n">10 lags</th>'
            f'<th class="n">5 lags + 4 grupos</th></tr>{linhas}</table>')


def t3_importancia():
    x0, x1, top = 140, W - 50, 30
    vmin, vmax = -0.2, 1.3
    X = lambda v: x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
    out = titulo("Quanto o erro piora ao embaralhar cada grupo (%)")
    for v in [0, 0.5, 1.0]:
        out += linha(X(v), top, X(v), top + 5 * 34, TINTA2 if v == 0 else GRADE, 1.2 if v == 0 else 1)
        out += txt(X(v), top + 5 * 34 + 14, f(v, 1) + "%", 10.5, TINTA2, "middle")
    for i, g in enumerate(imp.index):
        y = top + i * 34
        out += txt(x0 - 8, y + 19, g, 12, TINTA, "end", 650)
        for j, (c, cor) in enumerate([("60 min", LARANJA), ("15 min", CINZA)]):
            v = imp.loc[g, c]
            out += ret(min(X(0), X(v)), y + 5 + j * 13, abs(X(v) - X(0)), 11, cor)
            out += txt(X(max(v, 0)) + 5, y + 14 + j * 13, f(v, 2, True), 10, LARANJA if c == "60 min" else TINTA2, "start", 650)
    yl = top + 5 * 34 + 26
    out += ret(x0, yl - 9, 12, 10, LARANJA) + txt(x0 + 16, yl, "candle 60 min", 11, TINTA2)
    out += ret(x0 + 130, yl - 9, 12, 10, CINZA) + txt(x0 + 146, yl, "candle 15 min", 11, TINTA2)
    return svg(W, yl + 6, out)


def t4_envelhecimento():
    x0, x1, top, bot = 50, W - 60, 34, 170
    y_de = escala(48, 55, top, bot)
    out = titulo("Acerto de direção em cada mês após o treino, sem re-treinar")
    out += eixo_y(x0, x1, y_de, [48, 50, 52, 54], lambda v: f"{v:.0f}%")
    out += linha(x0, y_de(50), x1, y_de(50), TINTA2, 1, 'stroke-dasharray="2 3"')
    xs = [x0 + 30 + i * (x1 - x0 - 60) / 5 for i in range(6)]
    for c, cor in [("15 min", CINZA), ("60 min", LARANJA)]:
        pts = list(zip(xs, [y_de(v) for v in env_acerto[c]]))
        out += poli(pts, cor, 2.5) + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{cor}"/>' for x, y in pts)
        out += txt(pts[-1][0] + 8, pts[-1][1] + 4, c, 12, cor, "start", 700)
    for i, x in enumerate(xs):
        out += txt(x, bot + 16, f"{i + 1}º mês", 11, TINTA2, "middle")
    return svg(W, 192, out)


def t4_frequencia():
    x0, x1, top, bot = 50, W - 10, 34, 170
    y_de = escala(-2, 14, top, bot)
    out = titulo("Ganho bruto (pts/op) por frequência de re-treino · janela 24 meses")
    out += eixo_y(x0, x1, y_de, [0, 4, 8, 12])
    freqs = [("mensal", AZUL_MEDIO), ("trimestral", AZUL), ("semestral", TINTA)]
    gw = (x1 - x0) / 4
    for i, m in enumerate([5, 15, 30, 60]):
        for j, (fr, cor) in enumerate(freqs):
            v = g4.loc[(m, 24, fr), "ganho bruto pts/op"]
            bw = (gw - 40) / 3
            x = x0 + i * gw + 20 + j * bw
            out += ret(x, y_de(0), bw - 3, y_de(v) - y_de(0), cor)
            if m == 60:
                out += txt(x + bw / 2 - 1, y_de(v) - 5, f(v, 1), 10.5, TINTA, "middle", 650)
        out += txt(x0 + i * gw + gw / 2, bot + 16, f"{m} min", 11.5, TINTA2, "middle")
    for j, (fr, cor) in enumerate(freqs):
        out += ret(x0 + j * 105, bot + 27, 12, 10, cor) + txt(x0 + 16 + j * 105, bot + 36, fr, 11, TINTA2)
    return svg(W, 212, out)


def t5_heatmap():
    cands, jans = [5, 15, 30, 60], [3, 6, 12, 24, 36, 120]
    x0, y0, cw, ch = 64, 46, 98, 40
    out = titulo("Ganho bruto por operação (pts, sem custo) e t · re-treino trimestral")
    for j, jn in enumerate(jans):
        out += txt(x0 + j * cw + cw / 2, y0 - 7, "tudo" if jn == 120 else f"{jn} m", 11.5, TINTA2, "middle", 650)
    for i, m in enumerate(cands):
        out += txt(x0 - 8, y0 + i * ch + ch / 2 + 4, f"{m} min", 12, TINTA, "end", 650)
        for j, jn in enumerate(jans):
            r = g4.loc[(m, jn, "trimestral")]
            cor, escuro = heat_cor(r["ganho bruto pts/op"], 12)
            borda = f'stroke="{TINTA}" stroke-width="2.5"' if (m, jn) == (60, 36) else 'stroke="#fff" stroke-width="2"'
            out += ret(x0 + j * cw, y0 + i * ch, cw, ch, cor, borda)
            out += txt(x0 + j * cw + cw / 2, y0 + i * ch + 18, f(r["ganho bruto pts/op"], 1, True), 12.5, "#fff" if escuro else TINTA, "middle", 700)
            out += txt(x0 + j * cw + cw / 2, y0 + i * ch + 32, f"t = {f(r['t ganho bruto'], 1)} · {int(r['trimestres + (bruto)'])}/4",
                       10, "#fff" if escuro else TINTA2, "middle")
    out += txt(x0 + 3 * cw, y0 + 4 * ch + 16, "janela de treino (meses de histórico) · célula: ganho, t e trimestres positivos", 10.5, TINTA2, "middle")
    return svg(W, y0 + 4 * ch + 22, out)


def t5_r2():
    cands, jans = [5, 15, 30, 60], [3, 6, 12, 24, 36, 120]
    x0, x1, top, bot = 50, W - 110, 30, 140
    y_de = escala(-3, 0.5, top, bot)
    out = titulo("R² fora da amostra (%) × janela: mais histórico, menos ruído")
    out += eixo_y(x0, x1, y_de, [-3, -2, -1, 0])
    cores = {5: "#B9C6D6", 15: AZUL_MEDIO, 30: AZUL, 60: LARANJA}
    xs = [x0 + 20 + j * (x1 - x0 - 40) / 5 for j in range(6)]
    for m in cands:
        pts = list(zip(xs, [y_de(g4.loc[(m, jn, "trimestral"), "R² vs naive %"]) for jn in jans]))
        out += poli(pts, cores[m], 2.3) + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{cores[m]}"/>' for x, y in pts)
    for x, jn in zip(xs, jans):
        out += txt(x, bot + 15, "tudo" if jn == 120 else f"{jn} m", 10.5, TINTA2, "middle")
    for k, m in enumerate(cands):
        out += ret(x1 + 22, top + 6 + k * 18, 12, 10, cores[m]) + txt(x1 + 38, top + 15 + k * 18, f"{m} min", 11, TINTA2)
    return svg(W, 160, out)


def t6_tabela():
    rr = r5.loc["60 min × 36 m"]
    linhas = ""
    for k, rot in [("sempre", "opera sempre"), ("prev > 1×custo", "previsto > custo"), ("prev > 2×custo", "previsto > 2× custo"),
                   ("prev > 3×custo", "previsto > 3× custo"), ("top 20%", "20% mais fortes"), ("top 10%", "10% mais fortes")]:
        r = rr.loc[k]
        cls = ' class="foco"' if k == "prev > 1×custo" else ""
        linhas += (f"<tr{cls}><td>{rot}</td><td class='n'>{f(r['n_ops'], 0)}</td><td class='n'>{f(r['esperança pts/op'], 1, True)}</td>"
                   f"<td class='n'>{int(r['trimestres +'])}/4</td><td class='n'>{rs(r['total R$ (1 contrato)'])}</td>"
                   f"<td class='n'>{f(r['p vs aleatório'], 2)}</td></tr>")
    return (f'<div class="rotulo-res">60 min × 36 m · validação 2024 · já com 5 pts de custo por operação</div>'
            f'<table class="tab"><tr><th>Regra</th><th class="n">Operações</th><th class="n">pts/op</th><th class="n">Trim. +</th>'
            f'<th class="n">Total</th><th class="n">p vs aleat.</th></tr>{linhas}</table>')


def t6_curvas():
    x0, x1, top, bot = 50, W - 170, 28, 175
    ini, fim = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
    X = lambda t: x0 + (t - ini).total_seconds() / (fim - ini).total_seconds() * (x1 - x0)
    y_de = escala(-1000, 4500, top, bot)
    out = titulo("Capital acumulado em 2024 (R$, 1 contrato, com custo)")
    out += eixo_y(x0, x1, y_de, [-1000, 0, 1000, 2000, 3000, 4000], lambda v: f(v, 0))
    cores = {"opera sempre": CINZA, "previsto > custo": LARANJA, "previsto > 2× custo": AZUL_MEDIO, "previsto > 3× custo": AZUL}
    rot_y = []
    for rot, c in curvas5.items():
        c = c.resample("D").last().dropna()
        out += poli([(X(t), y_de(v)) for t, v in c.items()], cores[rot], 2.8 if rot == "previsto > custo" else 1.6)
        rot_y.append([y_de(c.iloc[-1]) + 4, rot])
    rot_y.sort()
    for k in range(1, len(rot_y)):
        rot_y[k][0] = max(rot_y[k][0], rot_y[k - 1][0] + 14)
    for y, rot in rot_y:
        out += txt(x1 + 6, y, rot, 11, cores[rot], "start", 700 if rot == "previsto > custo" else 500)
    for mth, r in [("2024-01-01", "jan"), ("2024-04-01", "abr"), ("2024-07-01", "jul"), ("2024-10-01", "out")]:
        out += txt(X(pd.Timestamp(mth)), bot + 15, r, 10.5, TINTA2, "middle")
    return svg(W, 194, out)


def t7_heatmap():
    prof, folha = [3, 4, 6, 8, 12], [0.001, 0.005, 0.01, 0.02, 0.05]
    x0, y0, cw, ch = 74, 44, 92, 34
    out = titulo("Esperança líquida (pts/op) · 50% das variáveis por divisão")
    for j, fl in enumerate(folha):
        out += txt(x0 + j * cw + cw / 2, y0 - 7, f(fl * 100, 1 if fl < 0.01 else 0) + "%", 11.5, TINTA2, "middle", 650)
    for i, p in enumerate(prof):
        out += txt(x0 - 8, y0 + i * ch + ch / 2 + 4, f"prof. {p}", 12, TINTA, "end", 650)
        for j, fl in enumerate(folha):
            v = g6.loc[(p, fl, 0.5), "esperança pts/op"]
            cor, escuro = heat_cor(v, 16)
            borda = f'stroke="{TINTA}" stroke-width="3"' if (p, fl, 0.5) == ESC else 'stroke="#fff" stroke-width="2"'
            out += ret(x0 + j * cw, y0 + i * ch, cw, ch, cor, borda)
            out += txt(x0 + j * cw + cw / 2, y0 + i * ch + 22, f(v, 1, True), 12.5, "#fff" if escuro else TINTA, "middle", 700)
    out += txt(x0 + 5 * cw + 8, y0 + ch + 22, "◀ escolhida", 12, TINTA, "start", 700)
    out += txt(x0 + 2.5 * cw, y0 + 5 * ch + 16, "folha mínima (% dos candles de treino)", 10.5, TINTA2, "middle")
    return svg(W, y0 + 5 * ch + 22, out)


def t7_profundidade():
    prof = [3, 4, 6, 8, 12]
    medias = g6.groupby(level=0)[["esperança pts/op", "R² %"]].mean()
    out = titulo("Média das 15 combinações de cada profundidade")
    o, y_de = barras(50, W - 10, 32, 140, [f"prof. {p}" for p in prof], medias["esperança pts/op"].values,
                     AZUL, 14, [0, 4, 8, 12], lambda v: f(v, 0), lambda v: f(v, 1, True),
                     sub=[f"R² {f(medias.loc[p, 'R² %'], 2, True)}%" for p in prof])
    return svg(W, 178, out + o)


def g_final():
    x0, x1, top, bot = 60, 600, 30, 230
    res = pd.concat([res_teste, res_extra])
    mov = np.concatenate([mov_teste, mov_extra])
    cap = res.cumsum() * rz.REAIS_POR_PONTO["WIN"]
    rng = np.random.default_rng(42)
    trajs = np.cumsum(rng.choice([-1, 1], size=(500, len(mov))) * mov - rz.CUSTO_PTS, axis=1) * rz.REAIS_POR_PONTO["WIN"]
    p5, p95 = np.percentile(trajs, 5, axis=0), np.percentile(trajs, 95, axis=0)
    vmin = np.floor(min(p5.min(), cap.min()) / 2000) * 2000
    vmax = np.ceil(max(p95.max(), cap.max()) / 2000) * 2000
    y_de = escala(vmin, vmax, top, bot)
    n = len(cap)
    X = lambda i: x0 + i / (n - 1) * (x1 - x0)
    out = titulo("Capital acumulado fora da amostra (R$, 1 contrato, com custo)")
    out += eixo_y(x0, x1, y_de, list(np.arange(np.ceil(vmin / 4000) * 4000, vmax + 1, 4000)), lambda v: f(v, 0))
    idx = np.linspace(0, n - 1, 300).astype(int)
    banda = " ".join(f"{X(i):.1f},{y_de(p95[i]):.1f}" for i in idx) + " " + " ".join(f"{X(i):.1f},{y_de(p5[i]):.1f}" for i in idx[::-1])
    out += f'<polygon points="{banda}" fill="{CINZA}" fill-opacity="0.3"/>'
    out += poli([(X(i), y_de(cap.iloc[i])) for i in idx], AZUL, 2.5)
    xs = X(len(res_teste))
    out += linha(xs, top, xs, bot, TINTA2, 1, 'stroke-dasharray="4 3"')
    out += txt((x0 + xs) / 2, bot + 18, "teste 01/2025 → 03/2026", 11.5, LARANJA, "middle", 700)
    out += txt((xs + x1) / 2, bot + 18, "extra 04 → 09/2026", 11.5, VERDE, "middle", 700)
    out += txt(x1 + 6, y_de(cap.iloc[-1]) + 4, rs(cap.iloc[-1]), 11.5, AZUL, "start", 700)
    out += ret(x0, bot + 32, 22, 10, CINZA, 'fill-opacity="0.3"') + txt(x0 + 28, bot + 41, "90% de 500 sorteios de direção nos mesmos candles", 11, TINTA2)
    out += linha(x0 + 360, bot + 37, x0 + 382, bot + 37, AZUL, 2.5) + txt(x0 + 388, bot + 41, "modelo", 11, TINTA2)
    return svg(W, bot + 48, out)


# ------------------------------------------------------------------ montagem dos slides
def fig(conteudo):
    return f'<figure class="fig">{conteudo}</figure>'


def slide_teste(passos, decisao, pesou, resultados, formula=""):
    """Slide de teste: esquerda = como foi feito + decisão; direita = resultados."""
    lis = "".join(f"<li>{p}</li>" for p in passos)
    pes = "".join(f"<li>{p}</li>" for p in pesou)
    form = f'<span class="formula">{formula}</span>' if formula else ""
    return (f'<div class="teste">'
            f'<div class="esq"><div class="bloco feito"><div class="rot">Como foi feito</div><ol>{lis}</ol>{form}</div>'
            f'<div class="bloco decisao"><div class="rot">Decisão</div><p class="dec">{decisao}</p>'
            f'<div class="rot2">O que pesou</div><ul>{pes}</ul></div></div>'
            f'<div class="dir"><div class="rot">Resultados</div>{resultados}</div></div>')


slides = []


def slide(kicker, titulo_, corpo, rodape, oculto=False):
    if not oculto:
        slides.append((kicker, titulo_, corpo, rodape))


ROD_VAL = "WIN · validação 2024"

# 1. Objetivo
slide("Frente de ML · Random Forest · objetivo",
      "Encontrar o cenário de treino da Random Forest que dá o maior retorno depois do custo",
      f"""
<div class="tres corpo">
  <div class="cartao"><h2>O que a árvore prevê</h2><p><strong>Quanto</strong> o preço do WIN sobe ou desce do candle <em>t</em> para o <em>t+1</em>:
  <span class="formula">100 · ln(P<sub>t+1</sub> ÷ P<sub>t</sub>)</span>Em %, convertido em pontos na hora de medir o dinheiro.</p></div>
  <div class="cartao"><h2>Treino "sem custo"</h2><p>A árvore só aprende a prever o movimento. O custo <strong>não entra no treino</strong>: entra depois, na regra que decide se a previsão vira operação.</p></div>
  <div class="cartao"><h2>Resultado em dinheiro</h2><p>Compra ou vende no fechamento de <em>t</em>, zera no de <em>t+1</em>. Custo de <strong>5 pts</strong> por operação (composição no slide da frente estatística); 1 ponto = R$ 0,20.</p></div>
</div>
{fig(g_fluxo())}
<p class="nota">Construído do zero, em 7 testes feitos nesta ordem: primeiro o que define o problema (candle, alvo, variáveis, histórico), depois a regra de operação e, por último, o ajuste fino da árvore.</p>""",
      "WIN$N · candles de 1 min · 15/10/2020 → 25/09/2026")

# 2. Divisão do tempo
slide("Divisão do tempo · acordada entre as frentes",
      "Todas as escolhas foram feitas em 2024; o teste foi rodado uma única vez",
      f"""
{fig(g_linha_tempo())}
<div class="tres corpo" style="margin-top:14px">
  <div class="metrica"><b>Validação · 2024</b><p>Onde comparamos os cenários e fazemos <strong>todas</strong> as escolhas: candle, janela, variáveis, regra e hiperparâmetros.</p></div>
  <div class="metrica"><b>Teste · 01/2025 → 03/2026</b><p>Rodado <strong>uma vez</strong>, com tudo congelado. Mostra o desempenho realista.</p></div>
  <div class="metrica"><b>Extra · 04 → 09/2026</b><p>Meses que nenhuma frente usou para nada: uma segunda prova limpa, só do WIN.</p></div>
</div>""",
      "Nunca embaralhamos: o modelo só aprende com o passado e é avaliado no futuro")

# Critério  (OCULTO temporariamente a pedido – trocar oculto=True por False para voltar)
slide("Método · como comparamos os cenários",
      "O melhor cenário é o que ganha depois do custo, de forma consistente",
      f"""
<div class="duas corpo">
  <div>
    <div class="metrica"><b>1 · Esperança por operação &gt; 0</b><p>Em média, cada operação ganha dinheiro, já descontados os 5 pts de custo.</p></div>
    <div class="metrica"><b>2 · Estatística t ≥ 2</b><p>O ganho médio é grande perto da sua variação: não é sorte de poucas operações.</p></div>
    <div class="metrica"><b>3 · Positivo em ≥ 3 dos 4 trimestres de 2024</b><p>O lucro não vem de um único período bom.</p></div>
    <div class="metrica"><b>4 · Comparação com o aleatório</b><p>Sorteamos a direção 1.000 vezes <strong>nos mesmos candles</strong> em que o modelo operou.</p></div>
  </div>
  <div>{fig(g_acerto_exemplo())}</div>
</div>""",
      "Critério definido antes de qualquer modelo", oculto=True)

# Teste 1 · candle
v1, v60 = viab[1], viab[60]
slide("Teste 1 de 7 · tamanho do candle · leitura dos dados, sem modelo",
      "Quanto maior o candle, maior o movimento e menor o peso do custo, mas há menos exemplos",
      slide_teste(
          [f"Período de treino (15/10/2020 → 12/2023, {n_pregoes_treino} pregões). <strong>Nenhum modelo treinado</strong>: só os dados.",
           "Juntamos os candles de 1 min em candles de 5, 15, 30 e 60 min.",
           "Para cada candle <em>t</em>: movimento = |fechamento(t+1) − fechamento(t)|, em pontos. Último candle do dia fica de fora. <strong>M</strong> = média.",
           "Usamos o custo de 5 pts como <strong>régua</strong>: quanto ele pesa perto de M e que acerto de direção um modelo precisaria para empatar."],
          "Levar <strong>5, 15, 30 e 60 min</strong> para os próximos testes; o <strong>1 min</strong> volta só como referência no Teste 5. O candle final não é escolhido aqui.",
          [f"Em 1 min o custo é {f(100 * 5 / v1['mov'], 0)}% do movimento: o modelo precisaria acertar {f(v1['pmin'], 1)}% só para empatar.",
           f"Candles maiores pagam o custo com folga ({f(v60['pmin'], 1)}% em 60 min), mas têm {v60['por_dia']:.0f} candles/dia contra {v1['por_dia']:.0f}: menos exemplos para a árvore aprender.",
           f"Comprar sempre, sem previsão, dá ≈ {f(v60['sempre'], 1)} pts/op: o resultado é só o custo."],
          fig(t1_movimento()) + fig(t1_acerto()),
          formula="p·M − (1−p)·M − 5 = 0  →  p = (1 + 5 ÷ M) ÷ 2"),
      "WIN · treino 15/10/2020 → 12/2023 · nenhum modelo treinado neste teste")

# Teste 2 · alvo
pa = por_ano
slide("Teste 2 de 7 · o alvo · o que a árvore aprende a prever",
      "A árvore prevê o retorno em %, porque o preço quase dobrou desde 2020",
      slide_teste(
          ["Medimos o movimento médio do candle de 15 min <strong>ano a ano</strong>, em pontos e em % do preço.",
           f"Olhamos a distribuição do retorno no treino: curtose = {f(curtose15, 1)} (numa normal seria 0).",
           "Medimos a autocorrelação: o retorno de <em>t</em> ajuda a prever o de <em>t+1</em>, … <em>t+5</em>? Comparamos com a faixa de ruído ±2/√n.",
           "Fixamos a referência a bater: prever retorno 0 (naive) tem R² = 0."],
          "Alvo = <strong>retorno em % do próximo candle</strong>; os pontos (% × preço) só são calculados para medir o dinheiro. Treino sem custo.",
          [f"Em pontos, o movimento vai de {f(pa['pts'].min(), 0)} a {f(pa['pts'].max(), 0)} conforme o nível do preço; em %, fica entre {f(pa['pct'].min(), 2)}% e {f(pa['pct'].max(), 2)}%.",
           "Caudas pesadas: alguns candles extremos podem dominar o treino (voltou no Teste 7).",
           "Autocorrelação dentro do ruído: o sinal, se existir, é fraco. R² acima de 0 já é bom."],
          fig(t2_anos()) + fig(t2_autocorr()),
          formula="alvo = 100 · ln(P(t+1) ÷ P(t))"),
      "WIN · candle 15 min · 2020 → 2026 (autocorrelação e curtose: treino até 2023)")

# Teste 3 · variáveis
slide("Teste 3 de 7 · variáveis · o que a árvore vê no candle t",
      "Ficamos com 5 lags e os quatro grupos de variáveis; o volume relativo é o mais usado",
      slide_teste(
          ["Uma RF <strong>fixa</strong> e moderada (300 árvores, profundidade 6, folha 0,5%), sem ajuste ainda.",
           "Treinada <strong>uma vez</strong> em 2020–2023, prevendo 2024. Alvo sem custo; medida = ganho bruto por operação (opera sempre na direção prevista).",
           "Comparamos 3, 5 e 10 lags (retornos dos últimos candles). Depois somamos 4 grupos: forma do candle, volatilidade, volume relativo e contexto do dia.",
           "Importância: embaralhamos cada grupo na validação e medimos quanto o erro piora."],
          "<strong>5 lags + os 4 grupos</strong> = 17 variáveis, para todos os testes seguintes.",
          ["5 lags foi melhor ou igual a 3 e 10 em 3 dos 4 candles.",
           "O conjunto completo foi o melhor ou perto do melhor em 15, 30 e 60 min (60 min: +12,9 pts brutos).",
           "Os números saltam muito entre colunas vizinhas (60 min: −2,8 → +7,1): não cortamos variável olhando uma célula. A RF tolera variáveis pouco úteis."],
          t3_tabela_lags() + fig(t3_importancia())),
      ROD_VAL + " · ganho bruto, sem custo · RF fixa treinada uma vez em 2020–2023")

# Teste 4 · re-treino
fr60 = {fr: g4.loc[(60, 24, fr), "ganho bruto pts/op"] for fr in ["mensal", "trimestral", "semestral"]}
slide("Teste 4 de 7 · frequência de re-treino",
      "O modelo não piorou com o tempo, então mantivemos o re-treino trimestral",
      slide_teste(
          ["<strong>Envelhecimento:</strong> treinamos a RF em 11 datas (início de cada trimestre de 2022 a 2024, com 12 meses de histórico).",
           "Sem re-treinar, medimos o acerto de direção em cada um dos 6 meses seguintes e tiramos a média das 11 datas (candles de 15 e 60 min).",
           "<strong>Comparação direta:</strong> na grade do Teste 5, cada combinação rodou 2024 com re-treino mensal, trimestral e semestral."],
          "Re-treino <strong>trimestral</strong> (o padrão acordado entre as frentes), agora com respaldo nos dados.",
          ["O acerto não cai nos meses seguintes ao treino: não há sinal de que o modelo \"envelheça\" em 6 meses.",
           f"As três frequências dão resultados parecidos (60 min × 24 m: {f(fr60['mensal'], 1)} / {f(fr60['trimestral'], 1)} / {f(fr60['semestral'], 1)} pts).",
           "Re-treinar mais vezes não trouxe ganho e custa mais tempo de processamento."],
          fig(t4_envelhecimento()) + fig(t4_frequencia())),
      "WIN · 2022 → 2024 · sem custo")

# Teste 5 · janela x candle
r6036 = g4.loc[(60, 36, "trimestral")]
r1 = g4.loc[(1, 12, "trimestral")]
slide("Teste 5 de 7 · janela de treino × candle",
      "Com mais histórico o modelo erra menos; 60 min com 36 meses teve o maior ganho",
      slide_teste(
          ["Simulamos 2024 como na vida real: a cada re-treino, a RF aprende <strong>só com os últimos N meses</strong> e prevê o período seguinte.",
           "Candles 5, 15, 30, 60 min × janelas de 3, 6, 12 e 24 meses × 3 frequências (+ 1 min como referência) = 52 combinações.",
           "A janela de 24 m ficou na borda da grade, então acrescentamos <strong>36 meses e todo o histórico</strong>: 60 combinações no total.",
           "Medida: ganho bruto por operação (sem custo), t e trimestres positivos. Candidata = t ≥ 2 e ≥ 3 de 4 trimestres positivos."],
          "<strong>60 min × 36 meses</strong>, re-treino trimestral (alternativa: 30 min × 36 m).",
          [f"Maior ganho bruto da grade: {f(r6036['ganho bruto pts/op'], 1, True)} pts/op, positivo nos 4 trimestres, R² ≥ 0; as janelas vizinhas (24 m, tudo) também foram boas.",
           f"Só o 1 min passou no t ≥ 2 (t = {f(r1['t ganho bruto'], 1)}), mas com {f(r1['ganho bruto pts/op'], 2)} pt por operação: sinal real, sem valor econômico.",
           f"Ressalva: o 60 min × 36 m tem t = {f(r6036['t ganho bruto'], 1)}, abaixo de 2. Escolhemos pela consistência."],
          fig(t5_heatmap()) + fig(t5_r2())),
      ROD_VAL + " · ganho bruto, sem custo")

# Teste 6 · regra
rr = r5.loc["60 min × 36 m"]
slide("Teste 6 de 7 · regra de operação · aqui entra o custo",
      "Só operar quando o movimento previsto paga o custo deu o maior lucro",
      slide_teste(
          ["Usamos as previsões de 2024 do cenário 60 min × 36 m (e de 30 min × 36 m como alternativa).",
           "Previsão em % → pontos: % × preço de fechamento.",
           "<strong>O custo entra pela primeira vez:</strong> resultado de cada operação = direção × movimento real − 5 pts.",
           "Comparamos: operar sempre; só se o |previsto| passar de 1×, 2× ou 3× o custo; só nas X% previsões mais fortes (limiar do trimestre anterior).",
           "Controle: 1.000 sorteios de direção nos mesmos candles."],
          "Operar só quando <strong>|movimento previsto| &gt; 5 pts</strong>, no candle de 60 min. O 30 min foi descartado.",
          [f"Maior lucro total ({rs(rr.loc['prev > 1×custo', 'total R$ (1 contrato)'])}), positiva nos 4 trimestres e direção melhor que o aleatório (p = {f(rr.loc['prev > 1×custo', 'p vs aleatório'], 2)}).",
           "É a regra sem número arbitrário: compara o tamanho previsto com o custo da operação.",
           "As regras \"X% mais fortes\" ganham mais por operação, mas operam pouco, perdem no 1º trimestre e somam menos.",
           "No 30 min, as regras ficaram indistinguíveis do aleatório."],
          t6_tabela() + fig(t6_curvas())),
      ROD_VAL + " · custo de 5 pts por operação")

# Teste 7 · hiperparâmetros
re_ = g6.loc[ESC]
canto = g6.loc[(3, 0.001, 0.33)]
slide("Teste 7 de 7 · hiperparâmetros da árvore",
      "Árvores rasas e folhas grandes funcionaram melhor; escolhemos profundidade 4 e folha 1%",
      slide_teste(
          ["Cenário fixo: 60 min × 36 m, re-treino trimestral, 17 variáveis, regra \"previsto &gt; custo\".",
           "Grade: profundidade (3, 4, 6, 8, 12) × folha mínima (0,1% a 5%) × fração de variáveis por divisão (33%, 50%, 100%) = 75 combinações.",
           "Medida: esperança líquida, trimestres positivos e total em R$.",
           "Para não escolher um pico de sorte: média de cada combinação com as vizinhas (±1 profundidade, ±1 folha)."],
          f"<strong>Profundidade 4, folha 1%, 50% das variáveis</strong>: {f(re_['esperança pts/op'], 1, True)} pts/op, {rs(re_['total R$ (1 contrato)'])}, 4/4 trimestres.",
          [f"Pela média da vizinhança houve empate com prof. 3 / folha 0,1% / 33% ({f(canto['esperança vizinhança'], 1)} × {f(re_['esperança vizinhança'], 1)} pts, dentro do ruído).",
           f"Desempate: a escolhida tem a vizinhança de maior lucro da grade ({rs(re_['total R$ vizinhança'])}) e fica no meio da região boa, não no canto.",
           "As 75 combinações deram lucro: o ajuste fino muda pouco; o que pesou foram os testes anteriores."],
          fig(t7_heatmap()) + fig(t7_profundidade())),
      ROD_VAL + " · custo de 5 pts por operação · regra \"previsto > custo\"")

# Resultado
v, te, ex = r7.iloc[:, 0], r7.iloc[:, 1], r7.iloc[:, 2]


def col(r, k, d=1, s=True):
    return f(r[k], d, s)


slide("Resultado · cenário congelado, fora da amostra",
      "No teste o lucro caiu para perto de zero: a vantagem de 2024 não se manteve",
      f"""<div class="duas larga-dir corpo">
<div>
<div class="caixa azul"><p><strong>Cenário escolhido:</strong> WIN · candle 60 min · janela 36 meses · re-treino trimestral · 17 variáveis · RF (300 árvores, prof. 4, folha 1%, 50% das variáveis) · treino sem custo · opera se o movimento previsto &gt; 5 pts.</p></div>
<table class="tab" style="margin-top:12px"><tr><th></th><th class="n">Validação 2024</th><th class="n">Teste 01/25–03/26</th><th class="n">Extra 04–09/26</th></tr>
<tr><td>Operações</td><td class="n">{f(v['n_ops'], 0)}</td><td class="n">{f(te['n_ops'], 0)}</td><td class="n">{f(ex['n_ops'], 0)}</td></tr>
<tr class="foco"><td><strong>Esperança líquida (pts/op)</strong></td><td class="n">{col(v, 'esperança pts/op')}</td><td class="n">{col(te, 'esperança pts/op')}</td><td class="n">{col(ex, 'esperança pts/op')}</td></tr>
<tr><td>Ganho bruto (pts/op)</td><td class="n">{col(v, 'ganho bruto pts/op (regra)')}</td><td class="n">{col(te, 'ganho bruto pts/op (regra)')}</td><td class="n">{col(ex, 'ganho bruto pts/op (regra)')}</td></tr>
<tr><td>t</td><td class="n">{col(v, 't', 1, False)}</td><td class="n">{col(te, 't', 1, False)}</td><td class="n">{col(ex, 't', 1, False)}</td></tr>
<tr><td>Trimestres positivos</td><td class="n">{int(v['trimestres +'])} de 4</td><td class="n">{int(te['trimestres +'])} de 5</td><td class="n">{int(ex['trimestres +'])} de 2</td></tr>
<tr><td>p vs aleatório</td><td class="n">{col(v, 'p vs aleatório', 2, False)}</td><td class="n">{col(te, 'p vs aleatório', 2, False)}</td><td class="n">{col(ex, 'p vs aleatório', 2, False)}</td></tr>
<tr><td>Total (1 contrato)</td><td class="n">{rs(v['total R$ (1 contrato)'])}</td><td class="n">{rs(te['total R$ (1 contrato)'])}</td><td class="n">{rs(ex['total R$ (1 contrato)'])}</td></tr>
</table>
<p class="nota">Por que caiu: escolhemos entre ~150 configurações olhando um único ano; o melhor resultado entre muitos sempre carrega sorte. Mesmo assim, no mesmo teste o modelo anterior da frente deu −4,8 pts/op.</p>
</div>
{fig(g_final())}
</div>""",
      "Teste e período extra rodados uma vez cada · custo de 5 pts por operação")

# ------------------------------------------------------------------ HTML
CSS = """
:root { --papel:#EEF2F0; --folha:#FBFCFB; --tinta:#12233D; --tinta-2:#4A5A6E; --grade:#C9D3D0; --janela:#DCE6F2;
        --azul:#1F5FA8; --laranja:#C0661A; --laranja-claro:#F6E6D8; --verde:#3F8F5A; }
* { box-sizing:border-box; margin:0; padding:0; }
html, body { height:100%; }
body { background:#D6DDDA; color:var(--tinta); font-family:"Archivo", system-ui, sans-serif; overflow:hidden; }
.palco { position:fixed; inset:0; display:grid; place-items:center; }
.slide { position:absolute; width:1280px; height:720px; background:var(--folha);
  background-image:linear-gradient(var(--papel) 1px, transparent 1px), linear-gradient(90deg, var(--papel) 1px, transparent 1px);
  background-size:40px 40px; padding:26px 50px 16px; display:none; flex-direction:column; box-shadow:0 2px 30px rgba(18,35,61,.18); }
.slide.ativo { display:flex; }
.kicker { font-size:12px; font-weight:650; color:var(--laranja); letter-spacing:.07em; text-transform:uppercase; margin-bottom:5px;
          font-family:"IBM Plex Mono", Consolas, monospace; }
h1 { font-size:28px; line-height:1.1; font-weight:760; font-variation-settings:"wdth" 116; letter-spacing:-.01em; max-width:1180px; }
h2 { font-size:17px; font-weight:700; margin-bottom:6px; }
p, li { font-size:14px; line-height:1.42; color:var(--tinta-2); }
strong { color:var(--tinta); font-weight:650; }
em { font-style:normal; color:var(--laranja); font-weight:650; }
.corpo { margin-top:16px; }
.duas { display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:start; }
.duas.larga-dir { grid-template-columns:1fr 1.15fr; }
.tres { display:grid; grid-template-columns:repeat(3, 1fr); gap:22px; align-items:start; }
.cartao { border-top:3px solid var(--tinta); padding-top:10px; }
.fig { background:var(--folha); border:1px solid var(--grade); padding:8px 12px 4px; margin-top:8px; }
.fig svg { display:block; width:100%; height:auto; font-family:"Archivo", system-ui, sans-serif; }
.fig svg text { font-variant-numeric:tabular-nums; }
.nota { font-size:12.5px; margin-top:10px; }
.formula { display:block; font-family:"IBM Plex Mono", Consolas, monospace; font-size:13px; color:var(--tinta); background:#F1F4F2;
           border:1px solid var(--grade); padding:5px 9px; margin:6px 0 2px; }
.caixa { padding:10px 14px; background:var(--laranja-claro); border-left:4px solid var(--laranja); }
.caixa p { color:var(--tinta); }
.caixa.azul { background:var(--janela); border-left-color:var(--azul); }
.metrica { border-top:1px solid var(--grade); padding:7px 0; }
.metrica b { color:var(--tinta); font-size:14.5px; }
.metrica p { font-size:13.5px; }
/* slides de teste */
.teste { display:grid; grid-template-columns:0.92fr 1.08fr; gap:22px; margin-top:12px; flex:1; min-height:0; }
.esq { display:flex; flex-direction:column; gap:10px; }
.rot, .rot2 { font-family:"IBM Plex Mono", Consolas, monospace; font-size:11.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.bloco { padding:9px 13px 10px; border:1px solid var(--grade); background:#fff; }
.bloco.feito { border-top:3px solid var(--azul); }
.bloco.feito .rot { color:var(--azul); margin-bottom:4px; }
.bloco.feito ol { padding-left:18px; }
.bloco.feito li { font-size:12.8px; line-height:1.36; margin-bottom:3px; }
.bloco.decisao { border-top:3px solid var(--laranja); background:#FFFBF7; }
.bloco.decisao .rot { color:var(--laranja); margin-bottom:3px; }
.bloco.decisao .dec { font-size:13.5px; color:var(--tinta); line-height:1.38; }
.bloco.decisao .rot2 { color:var(--tinta-2); margin:7px 0 2px; font-size:10.5px; }
.bloco.decisao ul { padding-left:16px; }
.bloco.decisao li { font-size:12.5px; line-height:1.34; margin-bottom:2px; }
.dir > .rot { color:var(--tinta-2); }
.rotulo-res { font-size:12.5px; font-weight:700; color:var(--tinta); margin:8px 0 2px; }
.tab { width:100%; border-collapse:collapse; font-size:12.5px; }
.tab th { text-align:left; font-size:11.5px; color:var(--tinta-2); font-weight:650; padding:4px 7px; border-bottom:2px solid var(--tinta); vertical-align:bottom; }
.tab td { padding:4px 7px; border-bottom:1px solid var(--grade); color:var(--tinta-2); font-variant-numeric:tabular-nums; }
.tab .n { text-align:right; }
.tab tr.foco td { background:var(--janela); color:var(--tinta); font-weight:650; }
.rodape { margin-top:auto; padding-top:7px; display:flex; justify-content:space-between; gap:20px; font-size:11px; color:var(--tinta-2);
          border-top:1px solid var(--grade); font-family:"IBM Plex Mono", Consolas, monospace; }
.nav { position:fixed; bottom:10px; left:50%; transform:translateX(-50%); display:flex; gap:8px; align-items:center;
       background:rgba(251,252,251,.92); border:1px solid var(--grade); padding:5px 10px; border-radius:6px; font-size:13px; z-index:5; }
.nav button { font:inherit; border:1px solid var(--grade); background:#fff; color:var(--tinta); padding:3px 10px; border-radius:4px; cursor:pointer; }
.nav button:hover { background:var(--janela); }
@media print {
  @page { size:1280px 720px; margin:0; }
  body { overflow:visible; background:#fff; }
  .palco { position:static; display:block; }
  .slide { position:relative; display:flex !important; transform:none !important; box-shadow:none; page-break-after:always; }
  .nav { display:none; }
}
"""

partes = []
for i, (k, t, c, r) in enumerate(slides, 1):
    partes.append(f'<section class="slide" aria-label="Slide {i}: {esc(t)}">'
                  f'<div class="kicker">{esc(k)}</div><h1>{esc(t)}</h1>{c}'
                  f'<div class="rodape"><span>Frente de ML · Random Forest · previsão T+1</span><span>{esc(r)}</span><span>{i} / {len(slides)}</span></div></section>')

HTML = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Random Forest do zero</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main class="palco">
{''.join(partes)}
</main>
<nav class="nav" aria-label="Navegação dos slides">
  <button type="button" id="ant" aria-label="Slide anterior">Anterior</button>
  <span id="contador" aria-live="polite">1 / {len(slides)}</span>
  <button type="button" id="prox" aria-label="Próximo slide">Próximo</button>
  <button type="button" id="tela" aria-label="Tela cheia">Tela cheia</button>
</nav>
<script>
  const slides = [...document.querySelectorAll(".slide")];
  const contador = document.getElementById("contador");
  let atual = 0;
  function escalar() {{
    const s = Math.min(window.innerWidth / 1320, (window.innerHeight - 50) / 750);
    slides.forEach(sl => sl.style.transform = `scale(${{s}})`);
  }}
  function mostrar(i) {{
    atual = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((sl, k) => sl.classList.toggle("ativo", k === atual));
    contador.textContent = `${{atual + 1}} / ${{slides.length}}`;
    history.replaceState(null, "", `#${{atual + 1}}`);
  }}
  document.getElementById("ant").onclick = () => mostrar(atual - 1);
  document.getElementById("prox").onclick = () => mostrar(atual + 1);
  document.getElementById("tela").onclick = () => {{
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.(); else document.exitFullscreen?.();
  }};
  document.addEventListener("keydown", e => {{
    if (["ArrowRight", "PageDown", " "].includes(e.key)) {{ e.preventDefault(); mostrar(atual + 1); }}
    if (["ArrowLeft", "PageUp"].includes(e.key)) {{ e.preventDefault(); mostrar(atual - 1); }}
    if (e.key === "Home") mostrar(0);
    if (e.key === "End") mostrar(slides.length - 1);
  }});
  window.addEventListener("resize", escalar);
  escalar(); mostrar((parseInt(location.hash.slice(1), 10) || 1) - 1);
</script>
</body>
</html>"""

with open(SAIDA, "w", encoding="utf-8") as fh:
    fh.write(HTML)
print("ok:", SAIDA, len(slides), "slides")
