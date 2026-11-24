"""Gera apresentacoes/review_frente_ml.html – review curto da frente de ML para o grupo.

Usa só arquivos já salvos em saidas/ (nada é treinado aqui):
  A3_*.csv, A4_previsoes.pkl, A4_envelhecimento.csv  -> Fase A (critério R², versão 2)
  passo06_previsoes.pkl, passo05_regras.csv, passo07*_resultado.csv -> versão 1 (critério lucro), marcados como tal
Rodar de dentro de frentes/rf_do_zero:  python gerar_review.py
"""
import html
import numpy as np
import pandas as pd
import rf_zero as rz

SAIDA = "../../apresentacoes/review_frente_ml.html"
AZUL, LARANJA, VERDE, VERMELHO, CINZA, TINTA, TINTA2, GRADE = (
    "#1F5FA8", "#C0661A", "#3F8F5A", "#B3423A", "#9AA6A2", "#12233D", "#4A5A6E", "#C9D3D0")
W = 620


def f(x, d=1, sinal=False):
    s = f"{x:+,.{d}f}" if sinal else f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def esc(s):
    return html.escape(str(s))


# ------------------------------------------------------------------ dados
grupos = pd.read_csv("saidas/A3_grupos.csv", index_col=[0, 1])
lags = pd.read_csv("saidas/A3_lags.csv", index_col=[0, 1])
imp = pd.read_csv("saidas/A3_importancia.csv", index_col=0)
p4 = pd.read_pickle("saidas/A4_previsoes.pkl")
g4 = pd.DataFrame({k: rz.resumo_acerto(v) for k, v in p4.items()}).T.astype(float)
g4.index.names = ["candle", "janela", "re-treino"]
cand4 = g4[g4["trimestres R²>0"] >= 3].sort_values("R² %", ascending=False)
esc4 = cand4.index[0]
faltam = [(1, 36, "trimestral"), (1, 120, "trimestral")]

# versão 1: hiperparâmetros (60 min × 36 m × trimestral, 5 lags) relidos pelo R²
p6 = pd.read_pickle("saidas/passo06_previsoes.pkl")
g6 = pd.DataFrame({k: rz.resumo_acerto(v) for k, v in p6.items()}).T.astype(float)
g6.index.names = ["prof", "folha", "mf"]
v1_lucro = pd.read_csv("saidas/passo06_grade.csv", index_col=[0, 1, 2])
r2_prof = g6.groupby(level="prof")["R² %"].mean()
r2_folha = g6.groupby(level="folha")["R² %"].mean()
melhor_r2_v1 = g6["R² %"].idxmax()

# versão 1: regra e resultados fora da amostra
r5 = pd.read_csv("saidas/passo05_regras.csv", index_col=[0, 1]).loc["60 min × 36 m"]
r7 = pd.read_csv("saidas/passo07b_resultado.csv", index_col=0)


# ------------------------------------------------------------------ helpers SVG
def svg(w, h, corpo):
    return f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg">{corpo}</svg>'


def txt(x, y, s, size=12, cor=TINTA2, anchor="start", peso=400):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{cor}" text-anchor="{anchor}" font-weight="{peso}">{esc(s)}</text>'


def linha(x1, y1, x2, y2, cor=GRADE, w=1, extra=""):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{cor}" stroke-width="{w}" {extra}/>'


def ret(x, y, w, h, cor, extra=""):
    if h < 0:
        y, h = y + h, -h
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{cor}" {extra}/>'


def escala(vmin, vmax, top, bot):
    return lambda v: bot - (v - vmin) / (vmax - vmin) * (bot - top)


def eixo(x0, x1, y_de, ticks, fmt):
    out = ""
    for v in ticks:
        out += linha(x0, y_de(v), x1, y_de(v), TINTA2 if v == 0 else GRADE, 1.3 if v == 0 else 1)
        out += txt(x0 - 6, y_de(v) + 4, fmt(v), 11, TINTA2, "end")
    return out


def barras(rotulos, valores, titulo, vmin, vmax, ticks, cores=None, sub=None, h=230, fmt_v=lambda v: f(v, 2, True)):
    x0, x1, top, bot = 50, W - 10, 32, h - (40 if sub else 26)
    y_de = escala(vmin, vmax, top, bot)
    out = txt(4, 15, titulo, 13, TINTA, "start", 700) + eixo(x0, x1, y_de, ticks, lambda v: f(v, 1))
    bw = (x1 - x0) / len(rotulos)
    for i, (r, v) in enumerate(zip(rotulos, valores)):
        c = cores[i] if cores else (VERDE if v > 0 else VERMELHO)
        x = x0 + i * bw + bw * 0.18
        out += ret(x, y_de(0), bw * 0.64, y_de(v) - y_de(0), c)
        out += txt(x + bw * 0.32, (y_de(v) - 5) if v >= 0 else (y_de(v) + 14), fmt_v(v), 11, TINTA, "middle", 700)
        out += txt(x + bw * 0.32, bot + 15, r, 11, TINTA2, "middle")
        if sub:
            out += txt(x + bw * 0.32, bot + 29, sub[i], 10, CINZA, "middle")
    return svg(W, h, out)


def heat_cor(v, lim):
    t = max(-1.0, min(1.0, v / lim))
    a = np.array([248, 249, 247])
    b = np.array([63, 143, 90]) if t >= 0 else np.array([179, 66, 58])
    c = (a + (b - a) * abs(t)).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})", abs(t) > 0.55


# ------------------------------------------------------------------ gráficos
def g_cenario():
    """R² (melhor frequência de cada célula) por candle × janela."""
    cands, jans = [1, 5, 15, 30, 60], [3, 6, 12, 24, 36, 120]
    x0, y0, cw, ch = 60, 44, 92, 38
    out = txt(4, 15, "R² fora da amostra (%) · melhor frequência de re-treino de cada célula", 13, TINTA, "start", 700)
    for j, jn in enumerate(jans):
        out += txt(x0 + j * cw + cw / 2, y0 - 7, "tudo" if jn == 120 else f"{jn} m", 11.5, TINTA2, "middle", 650)
    for i, m in enumerate(cands):
        out += txt(x0 - 8, y0 + i * ch + ch / 2 + 4, f"{m} min", 12, TINTA, "end", 650)
        for j, jn in enumerate(jans):
            x, y = x0 + j * cw, y0 + i * ch
            try:
                sub = g4.xs((m, jn), level=["candle", "janela"])
            except KeyError:
                sub = pd.DataFrame()
            if sub.empty:
                out += ret(x, y, cw, ch, "#F1F4F2", 'stroke="#fff" stroke-width="2"')
                out += txt(x + cw / 2, y + 23, "não rodou", 10.5, CINZA, "middle")
                continue
            fr = sub["R² %"].idxmax(); v = sub.loc[fr, "R² %"]
            cor, escuro = heat_cor(v, 0.6)
            borda = f'stroke="{TINTA}" stroke-width="2.5"' if (m, jn, fr) == esc4 else 'stroke="#fff" stroke-width="2"'
            out += ret(x, y, cw, ch, cor, borda)
            out += txt(x + cw / 2, y + 17, f(v, 2, True), 12, "#fff" if escuro else TINTA, "middle", 700)
            out += txt(x + cw / 2, y + 30, fr, 9.5, "#fff" if escuro else TINTA2, "middle")
    out += txt(x0 + 3 * cw, y0 + 5 * ch + 16, "janela de treino · borda escura = escolha provisória", 10.5, TINTA2, "middle")
    return svg(W, y0 + 5 * ch + 22, out)


def g_features():
    conj = ["+ lags", "+ forma", "+ volatilidade", "+ volume", "+ contexto_dia"]
    r2 = grupos["R² %"].unstack("candle")
    media = r2.mean(axis=1).loc[conj].values
    r60 = r2["60 min"].loc[conj].values
    rot = ["lags (10)", "+ forma", "+ volatil.", "+ volume", "+ contexto"]
    x0, x1, top, bot = 50, W - 10, 34, 180
    y_de = escala(-0.3, 0.5, top, bot)
    out = txt(4, 15, "R² (%) somando grupos de variáveis · validação 2024", 13, TINTA, "start", 700)
    out += eixo(x0, x1, y_de, [-0.2, 0, 0.2, 0.4], lambda v: f(v, 1))
    bw = (x1 - x0) / len(rot)
    for i in range(len(rot)):
        for k, (v, cor) in enumerate([(media[i], AZUL), (r60[i], LARANJA)]):
            x = x0 + i * bw + 14 + k * (bw - 28) / 2
            out += ret(x, y_de(0), (bw - 28) / 2 - 3, y_de(v) - y_de(0), cor)
            out += txt(x + (bw - 28) / 4, (y_de(v) - 4) if v >= 0 else (y_de(v) + 12), f(v, 2, True), 9.5, TINTA, "middle", 650)
        out += txt(x0 + i * bw + bw / 2, bot + 15, rot[i], 11, TINTA2, "middle")
    out += ret(x0, bot + 25, 12, 10, AZUL) + txt(x0 + 16, bot + 34, "média dos 5 candles (critério)", 11, TINTA2)
    out += ret(x0 + 220, bot + 25, 12, 10, LARANJA) + txt(x0 + 236, bot + 34, "candle 60 min", 11, TINTA2)
    return svg(W, bot + 42, out)


def g_importancia():
    v = imp["60 min"]
    return barras(list(v.index.str.replace("contexto_dia", "contexto")), list(v.values),
                  "Queda do R² ao embaralhar cada grupo · 60 min (pontos de R²)", 0, 0.8, [0, 0.2, 0.4, 0.6, 0.8],
                  cores=[AZUL] * len(v), h=190)


def g_hiper():
    rot = [f"prof. {p}" for p in r2_prof.index]
    a = barras(rot, list(r2_prof.values), "R² médio por profundidade (versão 1, prévia)", -0.7, 0.5, [-0.4, 0, 0.4], h=175)
    rotf = [f"folha {f(x * 100, 1 if x < 0.01 else 0)}%" for x in r2_folha.index]
    b = barras(rotf, list(r2_folha.values), "R² médio por folha mínima (versão 1, prévia)", -0.4, 0.5, [-0.4, 0, 0.4], h=175)
    return a, b


def g_regra():
    linhas = [("opera sempre", "sempre"), ("previsto > custo", "prev > 1×custo"), ("previsto > 2× custo", "prev > 2×custo"),
              ("20% mais fortes", "top 20%"), ("10% mais fortes", "top 10%")]
    return barras([a for a, _ in linhas], [r5.loc[b, "esperança pts/op"] for _, b in linhas],
                  "Esperança líquida (pts/op) por regra · validação 2024 · versão 1", 0, 40, [0, 10, 20, 30, 40],
                  cores=[CINZA, LARANJA, AZUL, AZUL, AZUL], sub=[f"{int(r5.loc[b, 'n_ops'])} ops" for _, b in linhas],
                  h=200, fmt_v=lambda v: f(v, 1, True))


def g_fora():
    vals = [r7.iloc[1, c] for c in range(3)]
    return barras(["validação 2024", "teste 01/25–03/26", "extra 04–09/26"], vals,
                  "Esperança líquida (pts/op) da versão 1 fora da amostra", -10, 16, [-10, 0, 10],
                  sub=[f"t = {f(r7.iloc[2, c], 1)}" for c in range(3)], h=190, fmt_v=lambda v: f(v, 1, True))


# ------------------------------------------------------------------ slides
def fig(c):
    return f'<figure class="fig">{c}</figure>'


def bloco(rot, itens, cls=""):
    lis = "".join(f"<li>{i}</li>" for i in itens)
    return f'<div class="bloco {cls}"><div class="rot">{rot}</div><ul>{lis}</ul></div>'


def status(s):
    cls = {"feito": "ok", "parcial": "parc", "pendente": "pend"}[s]
    return f'<span class="st {cls}">{s}</span>'


slides = []


def slide(kicker, titulo, corpo):
    slides.append((kicker, titulo, corpo))


m4, j4, f4 = esc4
r4 = g4.loc[esc4]

# 1. Visão geral
slide("Review da frente de ML · Random Forest · WIN",
      "Onde estamos: a RF foi refeita para escolher só por acerto; faltam hiperparâmetros e regra",
      f"""<div class="duas">
<div>
<div class="caixa azul"><p><strong>Como o trabalho está organizado agora</strong></p>
<p><strong>Fase A – RF "ingênua":</strong> candle, janela, re-treino, variáveis e hiperparâmetros escolhidos <strong>só pelo R² fora da amostra</strong> (quanto erra menos que prever zero; Theil U = √(1 − R²)). Nenhum custo, nenhum dinheiro.</p>
<p style="margin-top:6px"><strong>Fase B – lucro:</strong> com a RF congelada, entra o custo de 5 pts e ajustamos <strong>só a regra de entrada</strong>. Comparação de candles pelo lucro apenas no final, sem mudar a escolha.</p></div>
<p class="nota">Por que refazer: na versão 1 o custo e o lucro entraram em escolhas do modelo (candle, variáveis, hiperparâmetros). A versão 1 fica guardada em <code>frentes/rf_do_zero/v1_criterio_lucro/</code>.</p>
</div>
<table class="tab"><tr><th>Objetivo</th><th>Status</th><th>Resultado atual</th></tr>
<tr><td><strong>1 · Melhor cenário</strong><br><span class="peq">candle × janela × re-treino</span></td><td>{status("parcial")}</td><td>{m4} min × {"todo o histórico" if j4 == 120 else f"{j4} m"} × {f4} · R² {f(r4["R² %"], 2, True)}% (provisório)</td></tr>
<tr><td><strong>2 · Melhores features</strong></td><td>{status("feito")}</td><td>10 lags + forma + volatilidade + volume + contexto do dia (20 variáveis)</td></tr>
<tr><td><strong>3 · Hiperparâmetros</strong></td><td>{status("pendente")}</td><td>só a prévia da versão 1: árvores rasas e folhas grandes</td></tr>
<tr><td><strong>4 · Critérios de decisão</strong><br><span class="peq">regra de operação</span></td><td>{status("pendente")}</td><td>versão 1: "opera se previsto &gt; custo"; no teste caiu de +14 para +3 pts/op</td></tr>
</table></div>""")

# 2. Objetivo 1
slide("Objetivo 1 · melhor cenário · candle × janela × re-treino",
      f"Só o candle de 60 min com muito histórico erra menos que prever zero",
      f"""<div class="duas">
<div>
{bloco("O que foi feito", [
    "Simulação de 2024 com re-treino: a RF aprende só com os últimos N meses e prevê até o próximo re-treino.",
    "Candles 5, 15, 30, 60 min × janelas 3, 6, 12, 24, 36 m e todo o histórico × re-treino mensal, trimestral, semestral. 1 min só trimestral.",
    "Regra: maior R² entre as combinações com R² &gt; 0 em ≥ 3 dos 4 trimestres."], "feito")}
{bloco("Conclusões", [
    f"Só 4 combinações passam na regra, todas de 60 min com 36 m ou mais. Escolha provisória: <strong>{m4} min × {'todo o histórico' if j4 == 120 else f'{j4} m'} × {f4}</strong>, R² {f(r4['R² %'], 2, True)}% (U = {f(r4['Theil U'], 4)}).",
    "Janela é o que mais pesa: R² médio de −1,6% (3 m) a +0,06% (tudo). Com sinal fraco, a árvore precisa de muitos exemplos.",
    "A frequência de re-treino quase não muda o resultado (médias iguais); o modelo não mostrou envelhecimento claro."], "conc")}
{bloco("Pendente", ["1 min com 36 m e todo o histórico não rodou (sessão interrompida); o R² do 1 min sobe com a janela (−0,13 → −0,01)."], "pend")}
</div>
{fig(g_cenario())}
</div>""")

# 3. Objetivo 2
slide("Objetivo 2 · melhores features",
      "As 20 variáveis juntas dão o melhor R²; o volume relativo é a mais importante",
      f"""<div class="duas">
<div>
{bloco("O que foi feito", [
    "RF fixa (300 árvores, prof. 6, folha 0,5%) treinada uma vez em 2020–2023, prevendo 2024, nos 5 candles.",
    "Testamos 3, 5 e 10 lags; depois somamos 4 grupos um a um: forma do candle, volatilidade, volume relativo, contexto do dia.",
    "Regra: maior R² médio entre os 5 candles. Importância: queda do R² ao embaralhar cada grupo."], "feito")}
{bloco("Conclusões", [
    "<strong>10 lags</strong> (R² médio −0,095, contra −0,108 com 5 e −0,168 com 3).",
    "<strong>Todos os grupos</strong>: único conjunto com R² médio positivo (+0,05%). No 60 min chega a +0,44%.",
    "No 60 min o <strong>volume relativo</strong> é de longe o mais importante; no 1 min nenhum grupo importa."], "conc")}
{bloco("Ressalva", ["A média entre candles é puxada pelo 60 min (maiores valores). A escolha seria a mesma olhando só os candles com R² &gt; 0."], "pend")}
</div>
<div>{fig(g_features())}{fig(g_importancia())}</div>
</div>""")

# 4. Objetivo 3
pa, pb = g_hiper()
slide("Objetivo 3 · melhores hiperparâmetros",
      "Ainda não refeito com o R²; a prévia indica árvores rasas e folhas grandes",
      f"""<div class="duas">
<div>
{bloco("O que foi feito (versão 1)", [
    "Grade de 75 combinações: profundidade 3–12 × folha mínima 0,1–5% × variáveis por divisão 33/50/100%.",
    "Na versão 1 a escolha foi pelo <strong>lucro com custo</strong> (prof. 4, folha 1%, 50%) – critério que não vale mais.",
    f"Relendo a mesma grade pelo R²: melhor = prof. {melhor_r2_v1[0]}, folha {f(melhor_r2_v1[1] * 100, 0)}%, {int(melhor_r2_v1[2] * 100)}% das variáveis (R² {f(g6.loc[melhor_r2_v1, 'R² %'], 2, True)}%)."], "feito")}
{bloco("Conclusões (prévia)", [
    "Árvores rasas (prof. 3–4) têm R² positivo; profundas (8–12) ficam negativas: decoram ruído.",
    "Folha maior ajuda: cada previsão vira a média de muitos candles e amortece os extremos.",
    "É prévia: usa 5 lags e o cenário 60 min × 36 m × trimestral da versão 1."], "conc")}
{bloco("Pendente", ["Rodar o Passo 5 (já pronto) no cenário novo e com as 20 variáveis, escolhendo pelo R² médio na vizinhança."], "pend")}
</div>
<div>{fig(pa)}{fig(pb)}</div>
</div>""")

# 5. Objetivo 4
slide("Objetivo 4 · critérios de decisão · regra de operação",
      "Na versão 1, operar só quando o previsto paga o custo foi a melhor regra, mas não se sustentou fora da amostra",
      f"""<div class="duas">
<div>
{bloco("O que foi feito (versão 1)", [
    "Com a RF fixa, comparamos: operar sempre; só se o |movimento previsto| &gt; 1, 2 ou 3 × custo; só nas X% previsões mais fortes; e 1.000 sorteios de direção.",
    "Resultado = direção × movimento real − 5 pts por operação."], "feito")}
{bloco("Conclusões (versão 1)", [
    f"\"Previsto &gt; custo\": {f(r5.loc['prev > 1×custo', 'esperança pts/op'], 1, True)} pts/op e 4/4 trimestres em 2024; direção melhor que o aleatório (p = {f(r5.loc['prev > 1×custo', 'p vs aleatório'], 2)}).",
    "Filtrar demais (10–20% mais fortes) ganha mais por operação, mas opera pouco e oscila.",
    f"Fora da amostra caiu: teste {f(r7.iloc[1, 1], 1, True)} pts/op (t {f(r7.iloc[2, 1], 1)}), período extra {f(r7.iloc[1, 2], 1, True)} pts/op. A vantagem de 2024 era em boa parte sorte de seleção."], "conc")}
{bloco("Pendente", ["Rodar a Fase B (Passos 6 e 7, prontos) com a RF nova congelada. Regra de escolha: esperança &gt; 0 em ≥ 3 trimestres e maior lucro total."], "pend")}
</div>
<div>{fig(g_regra())}{fig(g_fora())}</div>
</div>""")

# 6. Próximos passos
slide("Próximos passos",
      "Terminar a Fase A, congelar a RF e só então medir o dinheiro",
      """<div class="tres">
<div class="cartao"><h2>1 · Fechar a Fase A</h2><ul class="lista">
<li>Rodar as 2 combinações que faltam do 1 min (36 m e todo o histórico).</li>
<li>Confirmar o cenário pela regra do R².</li>
<li>Rodar os hiperparâmetros (Passo 5) e congelar a RF.</li></ul></div>
<div class="cartao"><h2>2 · Fase B</h2><ul class="lista">
<li>Regra de operação com custo (Passo 6).</li>
<li>Comparação de candles pelo lucro, só para referência (Passo 7).</li>
<li>Sensibilidade ao custo: 2,5 / 5 / 7,5 / 12,5 pts, alinhada à frente estatística.</li></ul></div>
<div class="cartao"><h2>3 · Teste e dólar</h2><ul class="lista">
<li>Teste final uma única vez, com tudo congelado.</li>
<li>Repetir o roteiro inteiro para o WDO.</li>
<li>Comparar com as outras frentes pelo Theil U e pela esperança após o custo.</li></ul></div>
</div>""")

# 7. Dúvidas
slide("Dúvidas para o grupo",
      "Pontos que precisam de decisão conjunta",
      """<ol class="duvidas">
<li><strong>Janela "todo o histórico":</strong> venceu, mas é uma janela expansiva, não deslizante como no protocolo comum. Aceitamos?</li>
<li><strong>Re-treino semestral:</strong> venceu no 60 min por pouco, e as médias das três frequências são iguais. Seguimos a regra (maior R²) ou mantemos o trimestral acordado?</li>
<li><strong>Escolha de variáveis pela média dos candles:</strong> a média é puxada pelo 60 min. Melhor escolher variáveis por candle?</li>
<li><strong>R² de +0,5% (U = 0,998):</strong> é pouco, mas é o único caso que erra menos que prever zero. É suficiente para seguir à Fase B?</li>
<li><strong>Teste já visto:</strong> o período de teste e o extra já foram usados com o modelo antigo e com a versão 1. Como reportar o resultado da versão 2?</li>
<li><strong>Custo:</strong> mantemos 5 pts como base ou adotamos os três cenários da frente estatística (2,5 / 7,5 / 12,5)?</li>
</ol>""")

# ------------------------------------------------------------------ HTML
CSS = """
:root { --papel:#EEF2F0; --folha:#FBFCFB; --tinta:#12233D; --tinta-2:#4A5A6E; --grade:#C9D3D0; --janela:#DCE6F2;
        --azul:#1F5FA8; --laranja:#C0661A; --laranja-claro:#F6E6D8; --verde:#3F8F5A; --vermelho:#B3423A; }
* { box-sizing:border-box; margin:0; padding:0; }
html, body { height:100%; }
body { background:#D6DDDA; color:var(--tinta); font-family:"Archivo", system-ui, sans-serif; overflow:hidden; }
.palco { position:fixed; inset:0; display:grid; place-items:center; }
.slide { position:absolute; width:1280px; height:720px; background:var(--folha);
  background-image:linear-gradient(var(--papel) 1px, transparent 1px), linear-gradient(90deg, var(--papel) 1px, transparent 1px);
  background-size:40px 40px; padding:28px 50px 16px; display:none; flex-direction:column; box-shadow:0 2px 30px rgba(18,35,61,.18); }
.slide.ativo { display:flex; }
.kicker { font-size:12px; font-weight:650; color:var(--laranja); letter-spacing:.07em; text-transform:uppercase; margin-bottom:5px;
          font-family:"IBM Plex Mono", Consolas, monospace; }
h1 { font-size:27px; line-height:1.12; font-weight:760; font-variation-settings:"wdth" 116; letter-spacing:-.01em; max-width:1180px; margin-bottom:16px; }
h2 { font-size:16px; font-weight:700; margin-bottom:6px; }
p, li { font-size:13.5px; line-height:1.42; color:var(--tinta-2); }
strong { color:var(--tinta); font-weight:650; }
code { font-family:"IBM Plex Mono", Consolas, monospace; font-size:12px; }
.duas { display:grid; grid-template-columns:1fr 1.02fr; gap:24px; align-items:start; }
.tres { display:grid; grid-template-columns:repeat(3, 1fr); gap:22px; align-items:start; }
.cartao { border-top:3px solid var(--tinta); padding-top:10px; }
.lista { list-style:none; } .lista li { padding:7px 0; border-top:1px solid var(--grade); }
.fig { background:var(--folha); border:1px solid var(--grade); padding:8px 12px 4px; margin-bottom:10px; }
.fig svg { display:block; width:100%; height:auto; font-family:"Archivo", system-ui, sans-serif; }
.fig svg text { font-variant-numeric:tabular-nums; }
.caixa { padding:12px 14px; background:var(--laranja-claro); border-left:4px solid var(--laranja); }
.caixa.azul { background:var(--janela); border-left-color:var(--azul); }
.caixa p { color:var(--tinta); }
.nota { font-size:12.5px; margin-top:12px; }
.bloco { padding:8px 12px 9px; border:1px solid var(--grade); background:#fff; margin-bottom:9px; }
.bloco .rot { font-family:"IBM Plex Mono", Consolas, monospace; font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-bottom:3px; }
.bloco ul { padding-left:16px; } .bloco li { font-size:12.8px; line-height:1.36; margin-bottom:2px; }
.bloco.feito { border-top:3px solid var(--azul); } .bloco.feito .rot { color:var(--azul); }
.bloco.conc { border-top:3px solid var(--verde); } .bloco.conc .rot { color:var(--verde); } .bloco.conc li { color:var(--tinta); }
.bloco.pend { border-top:3px solid var(--laranja); background:#FFFBF7; } .bloco.pend .rot { color:var(--laranja); }
.tab { width:100%; border-collapse:collapse; font-size:13px; }
.tab th { text-align:left; font-size:11.5px; color:var(--tinta-2); font-weight:650; padding:6px 8px; border-bottom:2px solid var(--tinta); }
.tab td { padding:9px 8px; border-bottom:1px solid var(--grade); color:var(--tinta-2); vertical-align:top; }
.peq { font-size:11.5px; color:var(--tinta-2); }
.st { display:inline-block; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; padding:2px 8px; border-radius:3px; color:#fff; }
.st.ok { background:var(--verde); } .st.parc { background:var(--azul); } .st.pend { background:var(--laranja); }
.duvidas { padding-left:22px; max-width:1000px; } .duvidas li { font-size:15px; margin-bottom:12px; }
.rodape { margin-top:auto; padding-top:7px; display:flex; justify-content:space-between; font-size:11px; color:var(--tinta-2);
          border-top:1px solid var(--grade); font-family:"IBM Plex Mono", Consolas, monospace; }
.nav { position:fixed; bottom:10px; left:50%; transform:translateX(-50%); display:flex; gap:8px; align-items:center;
       background:rgba(251,252,251,.92); border:1px solid var(--grade); padding:5px 10px; border-radius:6px; font-size:13px; z-index:5; }
.nav button { font:inherit; border:1px solid var(--grade); background:#fff; color:var(--tinta); padding:3px 10px; border-radius:4px; cursor:pointer; }
@media print {
  @page { size:1280px 720px; margin:0; }
  body { overflow:visible; background:#fff; } .palco { position:static; display:block; }
  .slide { position:relative; display:flex !important; transform:none !important; box-shadow:none; page-break-after:always; }
  .nav { display:none; }
}
"""

partes = [f'<section class="slide" aria-label="Slide {i}: {esc(t)}"><div class="kicker">{esc(k)}</div><h1>{esc(t)}</h1>{c}'
          f'<div class="rodape"><span>Frente de ML · review interno</span><span>WIN · validação 2024 · Fase A pelo R², Fase B com custo de 5 pts</span>'
          f'<span>{i} / {len(slides)}</span></div></section>' for i, (k, t, c) in enumerate(slides, 1)]

HTML = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review frente ML</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<main class="palco">{''.join(partes)}</main>
<nav class="nav" aria-label="Navegação"><button type="button" id="ant">Anterior</button><span id="contador">1 / {len(slides)}</span>
<button type="button" id="prox">Próximo</button><button type="button" id="tela">Tela cheia</button></nav>
<script>
  const slides = [...document.querySelectorAll(".slide")]; const contador = document.getElementById("contador"); let atual = 0;
  function escalar() {{ const s = Math.min(window.innerWidth / 1320, (window.innerHeight - 50) / 750); slides.forEach(sl => sl.style.transform = `scale(${{s}})`); }}
  function mostrar(i) {{ atual = Math.max(0, Math.min(slides.length - 1, i)); slides.forEach((sl, k) => sl.classList.toggle("ativo", k === atual));
    contador.textContent = `${{atual + 1}} / ${{slides.length}}`; history.replaceState(null, "", `#${{atual + 1}}`); }}
  document.getElementById("ant").onclick = () => mostrar(atual - 1); document.getElementById("prox").onclick = () => mostrar(atual + 1);
  document.getElementById("tela").onclick = () => {{ if (!document.fullscreenElement) document.documentElement.requestFullscreen?.(); else document.exitFullscreen?.(); }};
  document.addEventListener("keydown", e => {{
    if (["ArrowRight", "PageDown", " "].includes(e.key)) {{ e.preventDefault(); mostrar(atual + 1); }}
    if (["ArrowLeft", "PageUp"].includes(e.key)) {{ e.preventDefault(); mostrar(atual - 1); }} }});
  window.addEventListener("resize", escalar); escalar(); mostrar((parseInt(location.hash.slice(1), 10) || 1) - 1);
</script></body></html>"""

with open(SAIDA, "w", encoding="utf-8") as fh:
    fh.write(HTML)
print("ok:", SAIDA, len(slides), "slides | cenário provisório:", esc4, "| melhor R² v1 hiper:", melhor_r2_v1)
