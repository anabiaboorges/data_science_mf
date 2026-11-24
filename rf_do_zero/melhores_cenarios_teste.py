"""Aplica em 2025-2026, sem reajuste, os melhores cenários de ML da validação 2024 (custo 5 pts, 1 contrato WIN).
Teste e extra já tinham sido vistos: isto é comparação, não serve para escolher um cenário novo.
Limiares dos classificadores = quantil da validação 2024, congelado. Saída: saidas/melhores_cenarios_teste.csv"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.join(AQUI, "..", "..")
sys.path += [AQUI, RAIZ, os.path.join(RAIZ, "frentes", "ml")]
import rf_zero as rz, tt1_comum as t, otim_rf as o

rng = np.random.default_rng(42)
ops = {}      # (cenário, período) -> DataFrame com bruto e pontos

# ---- A) regressor 60 min x 36 m, 5 lags (versão 1)
w = rz.carregar(f"{RAIZ}/dados/WINN_M1/WIN$N_M1.csv")
base, grupos = rz.criar_variaveis(rz.adicionar_alvo(rz.reamostrar(w, 60)), 5)
cols = rz.colunas(grupos, 5)
base = base.dropna(subset=cols + ["ret_prox"])
print("variáveis:", len(cols), "| dados até", w.index.max())
PER = {"val2024": rz.VALIDACAO, "teste 01/25-03/26": rz.TESTE, "extra 04-09/26": rz.EXTRA}
for nome, p, f, mf in [("RF60 maior lucro (6/1%/0,5)", 6, .01, .5), ("RF60 maior pts/op (3/0,5%/0,33)", 3, .005, .33),
                       ("RF60 congelada (4/1%/0,5)", 4, .01, .5)]:
    fab = lambda n: RandomForestRegressor(n_estimators=300, max_depth=p, min_samples_leaf=f, max_features=mf,
                                          n_jobs=-1, random_state=42)
    for per, iv in PER.items():
        d = rz.walk_forward(base, cols, iv, 36, "QS", fabrica=fab)
        pts = rz.prev_pontos(d); s = np.sign(pts).where(pts.abs() > 5, 0)
        b = (s * d["mov_prox"])[s != 0]
        ops[nome, per] = pd.DataFrame({"bruto": b, "pontos": b - 5})

# ---- B) classificadores de 3 classes, regra segurar
win = t.carregar(f"{RAIZ}/dados/WINN_M1/WIN$N_M1.csv"); wdo = t.carregar(f"{RAIZ}/dados/WDON_M1/WDO$N_M1.csv")
print("WDO até", wdo.index.max())
F = t.features()["COMPLETO"]; FX = F + sum(o.GRUPOS_NOVOS.values(), [])
for nome, c, jm, q, corte, feats in [("Clf 15min x 1m segurar 10%", "15min", 1, .90, 5., F),
                                     ("Clf 60min x 3m segurar 2%", "60min", 3, .98, 5., F),
                                     ("Clf 1min x 12m corte10 segurar 2%", "1min", 12, .98, 10., FX)]:
    prep = lambda d: o.rotular(o.adicionar_features(d) if feats is FX else d, corte)
    D = prep(t.montar_base(win, wdo, c)[0]); DW = prep(t.montar_base(win, win, c)[0])  # extra: WDO pode faltar
    lim = None
    for per, iv in PER.items():
        pr, _ = o.walk_forward_b(DW if per.startswith("extra") else D, feats, *iv, janela_meses=jm)
        if lim is None: lim = float(np.quantile(np.abs(pr.sinal), q))
        ops[nome, per] = o.ops_segurar(pr, lim)
    print(nome, "limiar", lim)

for nome in dict.fromkeys(k[0] for k in ops):
    ops[nome, "2025-2026 total"] = pd.concat([ops[nome, p] for p in list(PER)[1:]])
L = []
for (nome, per), d in ops.items():
    x, b = d.pontos, d.bruto.values
    r = dict(cenario=nome, periodo=per, n_ops=len(x))
    if len(x) > 1:
        ac = x.cumsum(); tr = x.groupby(x.index.to_period("Q")).sum()
        sim = np.array([(rng.choice([-1, 1], len(b)) * b).mean() - 5 for _ in range(1000)])
        r.update(bruto=b.mean(), liquido=x.mean(), t=x.mean() / (x.std() / len(x) ** .5), acerto=100 * (x > 0).mean(),
                 trim=f"{(tr > 0).sum()}/{len(tr)}", lucro_R=x.sum() * .2, dd_R=(ac.cummax().clip(lower=0) - ac).max() * .2,
                 p_aleat=(sim >= x.mean()).mean())
    L.append(r)
res = pd.DataFrame(L); res.to_csv(f"{AQUI}/saidas/melhores_cenarios_teste.csv", index=False)
print(res.round(2).to_string())
