"""Funções da fase de otimização do Random Forest (abordagem B: classificador de 3 classes).

Usado pelos notebooks frentes/ml/otimizacao_02_* em diante. Não altera o tt1_comum.py: reaproveita dele as bases,
as features, o calendário (validação/teste) e o valor do ponto.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import tt1_comum as t

CUSTO = 5.0                      # pts por operação completa
CORTE_CLASSE = 5.0               # sobe/desce = movimento maior que o custo
QUANTIS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98]
T_MIN, TRIM_MIN = 2.0, 3         # candidata: esperança > 0, t ≥ 2 e lucro em ≥ 3 de 4 trimestres
MARGEM_R = 150.0                 # margem por contrato WIN em R$ — A CONFIRMAR
RF_PADRAO = dict(n_estimators=100, max_depth=4, max_features=0.5, folha_frac=0.02, folha_min=10)


def rotular(df, corte=CORTE_CLASSE):
    """Acrescenta os pontos do candle seguinte e a classe (+1 sobe, 0 neutro, −1 desce)."""
    df = df.copy()
    df["mov_pts"] = df.close_prox - df.close
    df["classe"] = np.where(df.mov_pts > corte, 1, np.where(df.mov_pts < -corte, -1, 0))
    return df


def walk_forward_b(df, feats, inicio, fim, janela_meses, rf=RF_PADRAO, embargo=1, verbose=False):
    """Walk-forward trimestral (mesma lógica de tt1_comum.walk_forward) do classificador de 3 classes.
    P(sobe) e P(desce) = média, nas árvores, da fração de cada classe entre TODAS as linhas de treino da folha."""
    cortes = sorted(set(list(pd.date_range(inicio, fim, freq=t.RETREINO)) + [pd.Timestamp(fim)]))
    blocos, info = [], []
    for ini, fim_b in zip(cortes[:-1], cortes[1:]):
        tr = df[(df.index >= ini - pd.DateOffset(months=janela_meses)) & (df.index < ini)].iloc[:-embargo]
        te = df[(df.index >= ini) & (df.index < fim_b)]
        if te.empty or tr.empty:
            continue
        ml = max(rf["folha_min"], int(rf["folha_frac"] * len(tr)))
        Xtr, Xte = tr[feats].values, te[feats].values
        clf = RandomForestClassifier(n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
                                     max_features=rf["max_features"], min_samples_leaf=ml,
                                     n_jobs=-1, random_state=42).fit(Xtr, tr.classe.values)
        cls = tr.classe.values
        p_up, p_dn = np.zeros(len(te)), np.zeros(len(te))
        for est in clf.estimators_:
            f_tr, f_te = est.apply(Xtr), est.apply(Xte)
            k = est.tree_.node_count
            n = np.bincount(f_tr, minlength=k).astype(float)
            n = np.where(n > 0, n, 1)
            p_up += (np.bincount(f_tr, weights=(cls == 1), minlength=k) / n)[f_te]
            p_dn += (np.bincount(f_tr, weights=(cls == -1), minlength=k) / n)[f_te]
        nt = len(clf.estimators_)
        blocos.append(pd.DataFrame({"p_sobe": p_up / nt, "p_desce": p_dn / nt, "mov_pts": te.mov_pts,
                                    "close": te.close, "high_prox": te.high_prox, "low_prox": te.low_prox,
                                    "close_prox": te.close_prox, "volat": te.volat, "y_true": te.alvo},
                                   index=te.index))
        info.append(dict(inicio=ini.date(), n_treino=len(tr), folha_min=ml, n_prev=len(te)))
        if verbose:
            print(f"  {ini.date()}: treino {len(tr):,} (folha mín. {ml}) | prevê {len(te):,}")
    out = pd.concat(blocos)
    out["sinal"] = out.p_sobe - out.p_desce
    return out, pd.DataFrame(info)


def estat(x):
    x = pd.Series(x)
    if len(x) < 30:
        return dict(n_ops=len(x), esperanca_pts=np.nan, ep_pts=np.nan, t_stat=np.nan)
    ep = x.std() / np.sqrt(len(x))
    return dict(n_ops=len(x), esperanca_pts=x.mean(), ep_pts=ep, t_stat=x.mean() / ep if ep > 0 else np.nan)


def qualidade(prev, col="sinal"):
    """Qualidade do sinal sem custo e sem limiar: Spearman, acurácia e ganho bruto operando todo candle."""
    s = prev[col]
    bruto = np.sign(s) * prev.mov_pts
    mexeu = prev.mov_pts != 0
    return dict(spearman=s.corr(prev.mov_pts, method="spearman"),
                acc_dir=(np.sign(s) == np.sign(prev.mov_pts))[mexeu].mean(),
                **{f"bruto_{k}": v for k, v in estat(bruto[s != 0]).items()})


def _posicoes(sinal, dias, limiar):
    pos, p = np.zeros(len(sinal)), 0
    for i in range(len(sinal)):
        if i == 0 or dias[i] != dias[i - 1]:
            p = 0
        if sinal[i] > limiar:
            p = 1
        elif sinal[i] < -limiar:
            p = -1
        pos[i] = p
    return pos


def ops_segurar(prev, limiar, custo=CUSTO, col="sinal"):
    """Entra quando o sinal passa o limiar; mantém até passar o limiar do lado oposto; zera no fim do pregão."""
    dias = prev.index.date
    pos = _posicoes(prev[col].values, dias, limiar)
    nova = (pos != np.roll(pos, 1)) | (dias != np.roll(dias, 1))
    nova[0] = True
    d = pd.DataFrame({"op": np.cumsum(nova), "lado": pos, "pts": pos * prev.mov_pts.values, "inicio": prev.index})
    d = d[d.lado != 0]
    o = d.groupby("op").agg(lado=("lado", "first"), bruto=("pts", "sum"), candles=("pts", "size"),
                            inicio=("inicio", "first"))
    o["pontos"] = o.bruto - custo
    return o.set_index("inicio")


def ops_por_candle(prev, limiar, custo=CUSTO, col="sinal"):
    """Uma operação por candle liberado pela trava: entra no fechamento de t, sai no de t+1."""
    o = t.simular(prev.assign(y_pred=prev[col]), limiar, None, None, custo)
    o = o[o.lado != 0].copy()
    o["bruto"], o["candles"] = o.pontos + custo, 1
    return o


def aleatorio(o, custo=CUSTO, seed=42):
    """Mesmas operações e durações, direção sorteada."""
    rng = np.random.default_rng(seed)
    return o.assign(pontos=o.bruto * rng.choice([-1, 1], size=len(o)) - custo)


def avaliar(o, n_dias, margem=MARGEM_R):
    x = o.pontos
    r = estat(x)
    if len(x) == 0:
        return r
    acum = x.cumsum() * t.PONTO_WIN
    dd = float((acum.cummax().clip(lower=0) - acum).max())
    trim = x.groupby(x.index.to_period("Q")).sum()
    lucro = x.sum() * t.PONTO_WIN
    g, p = x[x > 0], x[x < 0]
    return dict(**r, ops_por_dia=len(x) / n_dias, candles_medios=o.candles.mean(), taxa_acerto=(x > 0).mean(),
                payoff=g.mean() / -p.mean() if len(g) and len(p) else np.nan,
                fator_lucro=g.sum() / -p.sum() if len(p) else np.inf,
                bruto_pts=o.bruto.mean(), lucro_R=lucro, drawdown_R=dd, retorno_capital=lucro / (margem + dd),
                trimestres_positivos=int((trim > 0).sum()), n_trimestres=len(trim))


def grade_regras(prev, quantis=QUANTIS, custo=CUSTO, col="sinal", com_aleatorio=True):
    """Todas as combinações regra × limiar sobre as previsões salvas (sem retreinar)."""
    n_dias = len(set(prev.index.date))
    linhas = []
    for q in quantis:
        lim = float(np.quantile(np.abs(prev[col]), q)) if q > 0 else 0.0
        for regra, f in [("por_candle", ops_por_candle), ("segurar", ops_segurar)]:
            o = f(prev, lim, custo, col)
            linhas.append(dict(regra=regra, limiar_q=q, limiar=lim, serie="modelo", **avaliar(o, n_dias)))
            if com_aleatorio and len(o):
                linhas.append(dict(regra=regra, limiar_q=q, limiar=lim, serie="aleatorio",
                                   **avaliar(aleatorio(o, custo), n_dias)))
    g = pd.DataFrame(linhas)
    g["candidata"] = ((g.serie == "modelo") & (g.esperanca_pts > 0) & (g.t_stat >= T_MIN)
                      & (g.trimestres_positivos >= TRIM_MIN))
    return g


def heatmap(tab, coluna, titulo, caminho, candles, fmt=".2f", menor_melhor=False):
    """Heatmap candle × janela (aceita qualquer lista de candles, inclusive 5 min)."""
    import matplotlib.pyplot as plt
    piv = tab.pivot(index="candle", columns="janela_meses", values=coluna).reindex(candles)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(piv.values, cmap="RdYlGn_r" if menor_melhor else "RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c} m" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "–" if pd.isna(v) else format(v, fmt), ha="center", va="center", fontsize=9)
    ax.set_xlabel("janela de treino"); ax.set_ylabel("candle"); ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


# ---------------- regras de saída (passo 3) ----------------
def posicoes_saida(prev, lim_entrada, saida="virar", frac_saida=0.0, max_candles=None, janela_horario=None,
                   col="sinal"):
    """Simula a posição candle a candle (decisão no fechamento de t, resultado no candle t+1).
    Entrada: |sinal| > lim_entrada (compra se positivo, venda se negativo), só dentro da janela de horário.
    Saída (a que vier primeiro):
      - fim do pregão (sempre) ou fim da janela de horário;
      - saida="virar": sinal passa lim_entrada do lado oposto (vira a mão);
        saida="enfraquecer": sinal do próprio lado cai abaixo de frac_saida × lim_entrada
        (frac 0 = sai quando o sinal troca de lado); se já passar o limiar oposto, vira a mão;
      - max_candles: tempo máximo na posição (não reentra no mesmo candle).
    Devolve o vetor de posições (+1/0/−1) e o id da operação de cada candle."""
    s = prev[col].values
    dias = prev.index.date
    hora = (prev.index.hour + prev.index.minute / 60).values
    n = len(s)
    pos, opid = np.zeros(n), np.zeros(n, dtype=int)
    p, dur, k = 0, 0, 0
    for i in range(n):
        novo_dia = i == 0 or dias[i] != dias[i - 1]
        dentro = janela_horario is None or (janela_horario[0] <= hora[i] < janela_horario[1])
        if novo_dia or not dentro:
            p, dur = 0, 0
        saiu_por_tempo = False
        if p != 0:
            if max_candles is not None and dur >= max_candles:
                p, dur, saiu_por_tempo = 0, 0, True
            elif saida == "virar":
                if p * s[i] < -lim_entrada:
                    p, dur = -p, 0
            else:  # enfraquecer
                if p * s[i] < -lim_entrada:
                    p, dur = -p, 0
                elif p * s[i] < frac_saida * lim_entrada:
                    p, dur = 0, 0
        if p == 0 and dentro and not saiu_por_tempo:
            if s[i] > lim_entrada:
                p, dur = 1, 0
            elif s[i] < -lim_entrada:
                p, dur = -1, 0
        if p != 0:
            if i == 0 or pos[i - 1] != p or novo_dia or dur == 0:
                k += 1
            dur += 1
        pos[i], opid[i] = p, k
    return pos, opid


def ops_saida(prev, lim_entrada, custo=CUSTO, **kw):
    pos, opid = posicoes_saida(prev, lim_entrada, **kw)
    d = pd.DataFrame({"op": opid, "lado": pos, "pts": pos * prev.mov_pts.values, "inicio": prev.index})
    d = d[d.lado != 0]
    o = d.groupby("op").agg(lado=("lado", "first"), bruto=("pts", "sum"), candles=("pts", "size"),
                            inicio=("inicio", "first"))
    o["pontos"] = o.bruto - custo
    return o.set_index("inicio")


# ---------------- variáveis novas (passo 5) ----------------
GRUPOS_NOVOS = {
    "multiescala": ["ret_acum5", "ret_acum15", "ret_acum30", "ret_acum60", "volat15", "volat60", "vol_rel15",
                    "amplitude15"],
    "estrutura_dia": ["dist_vwap", "pos_range_dia", "ret_dia", "gap_abertura", "max_dia_dist", "min_dia_dist"],
}


def adicionar_features(df):
    """Variáveis de contexto mais longo, todas conhecidas no fechamento do candle t (só usam t e o passado).
    Janelas em nº de candles (no candle de 1 min, 5 = 5 minutos)."""
    df = df.copy()
    dia = pd.Series(df.index.date, index=df.index)
    g = df.groupby(dia)
    # multiescala: retorno acumulado, volatilidade, volume e amplitude em janelas maiores
    for k in [5, 15, 30, 60]:
        df[f"ret_acum{k}"] = df["ret"].rolling(k, min_periods=1).sum()
    for k in [15, 60]:
        df[f"volat{k}"] = df["ret"].rolling(k, min_periods=2).std()
    df["vol_rel15"] = df["vol"] / df["vol"].rolling(15, min_periods=1).mean()
    df["amplitude15"] = df["amplitude"].rolling(15, min_periods=1).mean()
    # estrutura do dia: onde o preço está em relação ao próprio pregão
    pv = (df["close"] * df["vol"]).groupby(dia).cumsum()
    vwap = pv / df["vol"].groupby(dia).cumsum().replace(0, np.nan)
    df["dist_vwap"] = (df["close"] / vwap - 1) * 100
    max_dia, min_dia = g["high"].cummax(), g["low"].cummin()
    df["pos_range_dia"] = ((df["close"] - min_dia) / (max_dia - min_dia).replace(0, np.nan)).fillna(0.5)
    df["max_dia_dist"] = (max_dia / df["close"] - 1) * 100
    df["min_dia_dist"] = (df["close"] / min_dia - 1) * 100
    abertura = g["open"].transform("first")
    df["ret_dia"] = (df["close"] / abertura - 1) * 100
    fech_ant = g["close"].last().shift(1)
    df["gap_abertura"] = ((abertura / dia.map(fech_ant)) - 1) * 100
    return df.replace([np.inf, -np.inf], np.nan).dropna()
