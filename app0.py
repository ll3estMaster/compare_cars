from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


VehicleType = Literal["Combustao", "Hibrido", "Eletrico"]
PaymentType = Literal["A vista", "Financiado", "Assinatura"]


STATE_IPVA_DEFAULTS = {
    "AC": 2.0,
    "AL": 3.0,
    "AP": 3.0,
    "AM": 3.0,
    "BA": 2.5,
    "CE": 3.0,
    "DF": 3.5,
    "ES": 2.0,
    "GO": 3.75,
    "MA": 2.5,
    "MT": 3.0,
    "MS": 3.0,
    "MG": 4.0,
    "PA": 2.5,
    "PB": 2.5,
    "PR": 3.5,
    "PE": 3.0,
    "PI": 2.5,
    "RJ": 4.0,
    "RN": 3.0,
    "RS": 3.0,
    "RO": 3.0,
    "RR": 3.0,
    "SC": 2.0,
    "SP": 4.0,
    "SE": 3.0,
    "TO": 2.0,
}


@dataclass
class CarInputs:
    id: str
    nome: str
    carro_atual: bool
    tipo: VehicleType
    valor_base: float
    pagamento: PaymentType
    entrada_extra: float
    taxa_juros_am: float
    prazo_meses: int
    assinatura_mensal: float
    taxa_inicial_assinatura: float
    km_mes: float
    consumo_km_l: float
    consumo_km_kwh: float
    preco_combustivel: float
    preco_kwh: float
    percentual_recarga_externa: float
    ipva_percentual: float
    licenciamento_anual: float
    seguro_anual: float
    revisoes_anuais: tuple[float, ...]
    manutencao_anual: float
    pneus_anual: float
    estacionamento_anual: float
    outros_anuais: float
    horizonte_anos: int


def money(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short_money(value: float) -> str:
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{sign}R$ {abs_value / 1_000_000:.1f} mi".replace(".", ",")
    if abs_value >= 1_000:
        return f"{sign}R$ {abs_value / 1_000:.0f} mil"
    return money(value)


def cost_label(category: str, value: float) -> str:
    return f"{category}: {short_money(value)}" if value > 0 else ""


def monthly_money(value: float) -> str:
    return f"{money(value)}/mes"


def format_table_money(df: pd.DataFrame, money_columns: list[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in money_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(money)
    return formatted


def wrap_label(label: str, max_len: int = 16) -> str:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_len:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def car_assumptions(car: CarInputs, current_value: float, investment_return: float) -> list[tuple[str, str]]:
    capital_diff = 0 if car.carro_atual else car.valor_base - current_value
    items = [
        ("Tipo", car.tipo),
        ("Modalidade", car.pagamento),
        ("Valor base/IPVA", money(car.valor_base)),
        ("Dif. capital vs atual", money(capital_diff)),
        ("Km por mes", f"{car.km_mes:,.0f}".replace(",", ".")),
        ("IPVA", f"{car.ipva_percentual:.2f}% a.a.".replace(".", ",")),
        ("Seguro anual", money(car.seguro_anual)),
        ("Licenciamento anual", money(car.licenciamento_anual)),
        ("Manutencao anual", money(car.manutencao_anual)),
        ("Pneus anual", money(car.pneus_anual)),
        ("Estacionamento anual", money(car.estacionamento_anual)),
        ("Outros anual", money(car.outros_anuais)),
        ("Rendimento usado", f"{investment_return:.2f}% a.a.".replace(".", ",")),
    ]
    if car.tipo == "Eletrico":
        items.extend(
            [
                ("Consumo eletrico", f"{car.consumo_km_kwh:.2f} km/kWh".replace(".", ",")),
                ("Energia", money(car.preco_kwh).replace("R$ ", "R$ ") + "/kWh"),
            ]
        )
    elif car.tipo == "Hibrido":
        items.extend(
            [
                ("Consumo gerador", f"{car.consumo_km_l:.2f} km/l equiv.".replace(".", ",")),
                ("Combustivel", money(car.preco_combustivel).replace("R$ ", "R$ ") + "/l"),
                ("Recarga externa", f"{car.percentual_recarga_externa:.0f}%"),
            ]
        )
        if car.percentual_recarga_externa > 0:
            items.extend(
                [
                    ("Consumo eletrico", f"{car.consumo_km_kwh:.2f} km/kWh".replace(".", ",")),
                    ("Energia", money(car.preco_kwh).replace("R$ ", "R$ ") + "/kWh"),
                ]
            )
    else:
        items.extend(
            [
                ("Consumo", f"{car.consumo_km_l:.2f} km/l".replace(".", ",")),
                ("Combustivel", money(car.preco_combustivel).replace("R$ ", "R$ ") + "/l"),
            ]
        )
    if not car.carro_atual:
        if car.pagamento == "Financiado":
            items.extend(
                [
                    ("Entrada adicional", money(car.entrada_extra)),
                    ("Juros", f"{car.taxa_juros_am:.2f}% a.m.".replace(".", ",")),
                    ("Prazo", f"{car.prazo_meses} meses"),
                ]
            )
        if car.pagamento == "Assinatura":
            items.extend(
                [
                    ("Mensalidade", monthly_money(car.assinatura_mensal)),
                    ("Taxa inicial", money(car.taxa_inicial_assinatura)),
                ]
            )
    for year, value in enumerate(car.revisoes_anuais, start=1):
        items.append((f"Revisao ano {year}", money(value)))
    return items


def monthly_payment(principal: float, monthly_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    if monthly_rate <= 0:
        return principal / months
    return principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)


def energy_cost(car: CarInputs) -> float:
    annual_km = car.km_mes * 12
    external_charge_share = car.percentual_recarga_externa / 100 if car.tipo == "Hibrido" else 0

    if car.tipo == "Eletrico":
        return annual_km / max(car.consumo_km_kwh, 0.01) * car.preco_kwh

    if car.tipo == "Hibrido":
        generator_cost = annual_km * (1 - external_charge_share) / max(car.consumo_km_l, 0.01) * car.preco_combustivel
        plug_cost = annual_km * external_charge_share / max(car.consumo_km_kwh, 0.01) * car.preco_kwh
        return generator_cost + plug_cost

    return annual_km / max(car.consumo_km_l, 0.01) * car.preco_combustivel


def financing_cost_by_year(car: CarInputs, capital_difference: float) -> list[float]:
    if car.carro_atual or car.pagamento != "Financiado" or capital_difference <= 0:
        return [0.0] * car.horizonte_anos

    principal = max(capital_difference - car.entrada_extra, 0)
    payment = monthly_payment(principal, car.taxa_juros_am / 100, car.prazo_meses)
    total_interest = max(payment * car.prazo_meses - principal, 0)
    months_by_year = [min(12, max(car.prazo_meses - (year - 1) * 12, 0)) for year in range(1, car.horizonte_anos + 1)]
    financed_months_inside_horizon = sum(months_by_year)
    if financed_months_inside_horizon == 0:
        return [0.0] * car.horizonte_anos
    return [total_interest * months / financed_months_inside_horizon for months in months_by_year]


def simulate_car(car: CarInputs, current_value: float, investment_return: float) -> pd.DataFrame:
    if car.carro_atual:
        capital_difference = 0.0
    elif car.pagamento == "Assinatura":
        capital_difference = -current_value
    else:
        capital_difference = car.valor_base - current_value
    annual_opportunity = capital_difference * investment_return / 100
    financing_by_year = financing_cost_by_year(car, capital_difference)

    rows = []
    cumulative = 0.0

    for year in range(1, car.horizonte_anos + 1):
        revision = car.revisoes_anuais[year - 1] if year <= len(car.revisoes_anuais) else 0
        subscription = car.assinatura_mensal * 12 + (car.taxa_inicial_assinatura if year == 1 else 0)
        ipva = 0.0 if car.pagamento == "Assinatura" else car.valor_base * car.ipva_percentual / 100
        licenciamento = 0.0 if car.pagamento == "Assinatura" else car.licenciamento_anual
        seguro = 0.0 if car.pagamento == "Assinatura" else car.seguro_anual
        revision = 0.0 if car.pagamento == "Assinatura" else revision
        manutencao = 0.0 if car.pagamento == "Assinatura" else car.manutencao_anual
        pneus = 0.0 if car.pagamento == "Assinatura" else car.pneus_anual
        operating = (
            energy_cost(car)
            + ipva
            + licenciamento
            + seguro
            + revision
            + manutencao
            + pneus
            + car.estacionamento_anual
            + car.outros_anuais
            + subscription
        )
        total = operating + annual_opportunity + financing_by_year[year - 1]
        cumulative += total

        rows.append(
            {
                "Modelo": car.nome,
                "ID": car.id,
                "Ano": year,
                "Valor base IPVA": car.valor_base,
                "Diferenca de capital vs atual": capital_difference,
                "Combustivel/energia": energy_cost(car),
                "IPVA": ipva,
                "Licenciamento": licenciamento,
                "Seguro": seguro,
                "Revisoes": revision,
                "Manutencao": manutencao,
                "Pneus": pneus,
                "Estacionamento": car.estacionamento_anual,
                "Outros": car.outros_anuais,
                "Assinatura": subscription,
                "Custo de oportunidade": annual_opportunity,
                "Juros do financiamento": financing_by_year[year - 1],
                "Custo operacional": operating,
                "Custo anual comparavel": total,
                "Custo acumulado comparavel": cumulative,
            }
        )

    return pd.DataFrame(rows)


def add_general_assumptions() -> tuple[float, int, float]:
    st.subheader("Premissas gerais")
    col1, col2, col3 = st.columns(3)

    state = col1.selectbox("Estado", sorted(STATE_IPVA_DEFAULTS), index=sorted(STATE_IPVA_DEFAULTS).index("SP"))
    ipva_default = col1.number_input(
        "Aliquota padrao de IPVA (% a.a.)",
        min_value=0.0,
        max_value=10.0,
        value=float(STATE_IPVA_DEFAULTS[state]),
        step=0.1,
        help="Default editavel. Regras reais podem variar por UF, combustivel, ano e isencoes.",
    )
    horizon = col2.slider("Horizonte de comparacao (anos)", 1, 10, 5)
    investment_return = col3.number_input(
        "Rendimento liquido esperado (% a.a.)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
        help="Usado para calcular o custo de oportunidade do dinheiro extra colocado na troca.",
    )
    return ipva_default, horizon, investment_return


def car_form(index: int, ipva_default: float, horizon: int, current_value: float | None = None) -> CarInputs:
    current = index == 0
    label = "Carro atual" if current else f"Modelo {index}"

    with st.expander(label, expanded=index <= 2):
        col1, col2, col3 = st.columns(3)

        with col1:
            nome = st.text_input("Nome", value=label, key=f"name_{index}")
            pagamento = "A vista"
            if not current:
                pagamento = st.selectbox("Modalidade", ["A vista", "Financiado", "Assinatura"], key=f"pay_{index}")
            tipo = st.selectbox("Tipo", ["Combustao", "Hibrido", "Eletrico"], key=f"type_{index}")
            valor_default = 80_000.0 if current else 120_000.0
            valor_label = "Valor de mercado hoje" if current else "Valor do modelo"
            if pagamento == "Assinatura":
                valor_label = "Valor de referencia do modelo"
            valor_base = st.number_input(
                valor_label,
                0.0,
                2_000_000.0,
                valor_default,
                1_000.0,
                key=f"value_{index}",
                help="Na compra, usado para IPVA e capital adicional. Na assinatura, e apenas referencia do modelo.",
            )
            km_mes = st.number_input("Km por mes", 0.0, 20_000.0, 1_200.0, 100.0, key=f"km_{index}")
            ipva = 0.0
            if pagamento != "Assinatura":
                ipva = st.number_input("IPVA (% a.a.)", 0.0, 10.0, ipva_default, 0.1, key=f"ipva_{index}")
            if not current and current_value is not None:
                if pagamento == "Assinatura":
                    st.info(f"Capital liberado ao vender o carro atual: {money(current_value)}")
                else:
                    diff = valor_base - current_value
                    if diff >= 0:
                        st.info(f"Capital adicional vs carro atual: {money(diff)}")
                    else:
                        st.info(f"Capital liberado vs carro atual: {money(abs(diff))}")

        with col2:
            consumo_l = 11.0
            consumo_kwh = 6.0
            preco_combustivel = 6.00
            preco_kwh = 0.95
            percentual_recarga_externa = 0.0

            if tipo == "Combustao":
                consumo_l = st.number_input("Consumo (km/l)", 0.1, 100.0, 11.0, 0.5, key=f"km_l_{index}")
                preco_combustivel = st.number_input("Combustivel (R$/l)", 0.0, 20.0, 6.00, 0.05, key=f"fuel_price_{index}")

            if tipo == "Hibrido":
                consumo_l = st.number_input(
                    "Consumo no modo gerador (km/l equiv.)",
                    0.1,
                    100.0,
                    16.0,
                    0.5,
                    key=f"km_l_{index}",
                    help="Use o consumo real quando a bateria e sustentada pelo motor gerador.",
                )
                preco_combustivel = st.number_input("Combustivel (R$/l)", 0.0, 20.0, 6.00, 0.05, key=f"fuel_price_{index}")
                percentual_recarga_externa = st.slider(
                    "% da energia carregada na tomada",
                    0.0,
                    100.0,
                    0.0,
                    5.0,
                    key=f"external_charge_{index}",
                )

            if tipo == "Eletrico" or (tipo == "Hibrido" and percentual_recarga_externa > 0):
                consumo_kwh = st.number_input("Consumo (km/kWh)", 0.1, 20.0, 6.0, 0.1, key=f"km_kwh_{index}")
                preco_kwh = st.number_input("Energia (R$/kWh)", 0.0, 5.0, 0.95, 0.05, key=f"kwh_price_{index}")

        with col3:
            entrada_extra = 0.0
            taxa = 0.0
            prazo = 0
            assinatura_mensal = 0.0
            taxa_inicial_assinatura = 0.0

            if pagamento == "Financiado":
                entrada_extra = st.number_input("Entrada sobre capital adicional", 0.0, 2_000_000.0, 0.0, 1_000.0, key=f"down_{index}")
                taxa = st.number_input("Juros (% a.m.)", 0.0, 10.0, 1.3, 0.05, key=f"rate_{index}")
                prazo = st.number_input("Prazo (meses)", 1, 120, 48, 6, key=f"term_{index}")

            if pagamento == "Assinatura":
                assinatura_mensal = st.number_input("Mensalidade da assinatura", 0.0, 100_000.0, 3_500.0, 100.0, key=f"sub_month_{index}")
                taxa_inicial_assinatura = st.number_input("Taxa inicial/adesao", 0.0, 100_000.0, 0.0, 500.0, key=f"sub_fee_{index}")
                seguro = 0.0
                licenciamento = 0.0
                st.info("IPVA, seguro, licenciamento, revisoes, manutencao e pneus foram considerados inclusos na assinatura.")
            else:
                seguro = st.number_input("Seguro anual", 0.0, 100_000.0, valor_base * 0.035, 250.0, key=f"ins_{index}")
                licenciamento = st.number_input("Licenciamento anual", 0.0, 10_000.0, 250.0, 50.0, key=f"lic_{index}")

        if pagamento == "Assinatura":
            st.caption("Custos fora do contrato")
            c1, c2 = st.columns(2)
            manutencao = 0.0
            pneus = 0.0
            estacionamento = c1.number_input("Estacionamento/ano", 0.0, 100_000.0, 0.0, 100.0, key=f"park_{index}")
            outros = c2.number_input("Outros/ano", 0.0, 100_000.0, 0.0, 100.0, key=f"other_{index}")
            revisoes = tuple(0.0 for _ in range(horizon))
        else:
            st.caption("Custos de manutencao e uso")
            c1, c2, c3, c4 = st.columns(4)
            manutencao = c1.number_input("Manutencao/ano", 0.0, 100_000.0, 1_000.0, 100.0, key=f"maint_{index}")
            pneus = c2.number_input("Pneus/ano", 0.0, 100_000.0, 800.0, 100.0, key=f"tires_{index}")
            estacionamento = c3.number_input("Estacionamento/ano", 0.0, 100_000.0, 0.0, 100.0, key=f"park_{index}")
            outros = c4.number_input("Outros/ano", 0.0, 100_000.0, 0.0, 100.0, key=f"other_{index}")

            st.caption("Revisoes por ano")
            revision_columns = st.columns(min(horizon, 5))
            revisoes = []
            for year in range(1, horizon + 1):
                column = revision_columns[(year - 1) % len(revision_columns)]
                revisoes.append(
                    column.number_input(
                        f"Ano {year}",
                        0.0,
                        100_000.0,
                        1_200.0 + 250.0 * ((year - 1) % 3),
                        100.0,
                        key=f"rev_{index}_{year}",
                    )
                )

    return CarInputs(
        id=f"car_{index}",
        nome=nome,
        carro_atual=current,
        tipo=tipo,
        valor_base=valor_base,
        pagamento=pagamento,
        entrada_extra=entrada_extra,
        taxa_juros_am=taxa,
        prazo_meses=int(prazo),
        assinatura_mensal=assinatura_mensal,
        taxa_inicial_assinatura=taxa_inicial_assinatura,
        km_mes=km_mes,
        consumo_km_l=consumo_l,
        consumo_km_kwh=consumo_kwh,
        preco_combustivel=preco_combustivel,
        preco_kwh=preco_kwh,
        percentual_recarga_externa=percentual_recarga_externa,
        ipva_percentual=ipva,
        licenciamento_anual=licenciamento,
        seguro_anual=seguro,
        revisoes_anuais=tuple(revisoes),
        manutencao_anual=manutencao,
        pneus_anual=pneus,
        estacionamento_anual=estacionamento,
        outros_anuais=outros,
        horizonte_anos=horizon,
    )


def make_break_even(current: CarInputs, target: CarInputs, investment_return: float) -> pd.DataFrame:
    target_prices = np.linspace(max(target.valor_base * 0.55, 10_000), target.valor_base * 1.45, 36)
    consumptions = np.linspace(5, 25, 36) if target.tipo != "Eletrico" else np.linspace(3, 10, 36)
    current_cost = simulate_car(current, current.valor_base, investment_return)["Custo acumulado comparavel"].iloc[-1]
    rows = []

    for price in target_prices:
        for consumption in consumptions:
            candidate = CarInputs(**target.__dict__)
            candidate.valor_base = float(price)
            candidate.seguro_anual = price * (target.seguro_anual / max(target.valor_base, 1))
            if candidate.tipo == "Eletrico":
                candidate.consumo_km_kwh = float(consumption)
            else:
                candidate.consumo_km_l = float(consumption)
            total = simulate_car(candidate, current.valor_base, investment_return)["Custo acumulado comparavel"].iloc[-1]
            rows.append(
                {
                    "Valor do carro": price,
                    "Consumo": consumption,
                    "Diferenca vs atual": total - current_cost,
                }
            )

    return pd.DataFrame(rows)


def add_pdf_table(pdf: PdfPages, title: str, df: pd.DataFrame, money_columns: list[str]) -> None:
    import matplotlib.pyplot as plt

    display_df = format_table_money(df, money_columns)
    display_df = display_df.rename(columns={column: wrap_label(column) for column in display_df.columns})
    rows_per_page = 16

    for start in range(0, len(display_df), rows_per_page):
        page_df = display_df.iloc[start : start + rows_per_page]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title if start == 0 else f"{title} (cont.)", fontsize=14, fontweight="bold", pad=16)
        table = ax.table(
            cellText=page_df.values,
            colLabels=page_df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.55)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#E8EEF7")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_pdf_table_blocks(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    money_columns: list[str],
    fixed_columns: list[str],
    columns_per_block: int = 5,
) -> None:
    variable_columns = [column for column in df.columns if column not in fixed_columns]
    for start in range(0, len(variable_columns), columns_per_block):
        block_columns = fixed_columns + variable_columns[start : start + columns_per_block]
        suffix = "" if start == 0 else f" - bloco {start // columns_per_block + 1}"
        add_pdf_table(pdf, title + suffix, df[block_columns], [col for col in money_columns if col in block_columns])


def add_pdf_assumptions(
    pdf: PdfPages,
    cars: list[CarInputs],
    current_value: float,
    investment_return: float,
    horizon: int,
) -> None:
    import matplotlib.pyplot as plt

    for car in cars:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.02, 0.96, f"Premissas imputadas - {car.nome}", fontsize=16, fontweight="bold")
        ax.text(0.02, 0.92, f"Horizonte: {horizon} anos", fontsize=10)

        items = car_assumptions(car, current_value, investment_return)
        columns = 3
        rows_per_column = int(np.ceil(len(items) / columns))
        for col in range(columns):
            x = 0.02 + col * 0.32
            y = 0.86
            for label, value in items[col * rows_per_column : (col + 1) * rows_per_column]:
                ax.text(x, y, label, fontsize=8, color="#4B5563")
                ax.text(x, y - 0.025, value, fontsize=9, fontweight="bold")
                y -= 0.065
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_pdf_report(
    cars: list[CarInputs],
    current: CarInputs,
    winner: pd.Series,
    final: pd.DataFrame,
    comparison_vs_current: pd.DataFrame,
    detailed_summary: pd.DataFrame,
    simulations: pd.DataFrame,
    breakdown: pd.DataFrame,
    investment_return: float,
    horizon: int,
) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = BytesIO()
    money_columns = [column for column in detailed_summary.columns if column != "Modelo"]

    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.02, 0.94, "Relatorio de comparacao de carros", fontsize=20, fontweight="bold")
        ax.text(0.02, 0.87, f"Carro atual: {current.nome}", fontsize=12)
        ax.text(0.02, 0.83, f"Horizonte: {horizon} anos", fontsize=12)
        ax.text(0.02, 0.79, f"Rendimento liquido esperado: {investment_return:.1f}% a.a.", fontsize=12)
        ax.text(0.02, 0.73, f"Melhor opcao: {winner['Modelo']}", fontsize=14, fontweight="bold")
        ax.text(0.02, 0.69, f"Custo acumulado comparavel: {money(winner['Custo acumulado comparavel'])}", fontsize=12)
        ax.text(0.02, 0.65, f"Custo mensal efetivo: {money(winner['Custo mensal efetivo'])}/mes", fontsize=12)
        ax.text(
            0.02,
            0.58,
            "Criterio: soma de custos de uso, assinatura quando aplicavel, IPVA, seguro, manutencao, "
            "revisoes, custo de oportunidade e juros do financiamento quando aplicavel.",
            fontsize=11,
            wrap=True,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        add_pdf_assumptions(pdf, cars, current.valor_base, investment_return, horizon)

        add_pdf_table(
            pdf,
            "Comparacao de cada modelo contra o atual",
            comparison_vs_current,
            [
                "Diferenca vs carro atual",
                "Diferenca mensal vs atual",
                "Economia vs atual",
                "Custo mensal efetivo",
                "Custo acumulado comparavel",
            ],
        )

        add_pdf_table_blocks(
            pdf,
            "Custos acumulados por categoria",
            detailed_summary,
            money_columns,
            fixed_columns=["Modelo"],
            columns_per_block=5,
        )

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        for model, rows in simulations.groupby("Modelo"):
            ax.plot(rows["Ano"], rows["Custo acumulado comparavel"], marker="o", label=model)
            for _, row in rows.iterrows():
                ax.annotate(
                    short_money(row["Custo acumulado comparavel"]),
                    (row["Ano"], row["Custo acumulado comparavel"]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                )
        ax.set_title("Custo acumulado comparavel ao longo do tempo", fontweight="bold")
        ax.set_xlabel("Ano")
        ax.set_ylabel("Custo acumulado")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        monthly = final.set_index("Modelo")["Custo mensal efetivo"].sort_values()
        bars = ax.bar(monthly.index, monthly.values)
        ax.set_title("Custo mensal efetivo por modelo", fontweight="bold")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Custo mensal")
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(bars, labels=[monthly_money(value) for value in monthly.values], padding=4)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_breakdown = breakdown.set_index("Modelo")
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        plot_breakdown.plot(kind="bar", stacked=True, ax=ax)
        totals = plot_breakdown.sum(axis=1)
        for index, total in enumerate(totals):
            ax.text(index, total, short_money(total), ha="center", va="bottom", fontsize=8, fontweight="bold")
        for category, container in zip(plot_breakdown.columns, ax.containers):
            labels = [short_money(value) if value > 0 else "" for value in container.datavalues]
            ax.bar_label(container, labels=labels, label_type="center", fontsize=6)
        ax.set_title("Composicao do custo acumulado", fontweight="bold")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Custo")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        category_totals = breakdown.copy()
        category_totals["Total"] = category_totals.drop(columns=["Modelo"]).sum(axis=1)
        add_pdf_table_blocks(
            pdf,
            "Valores por categoria usados no grafico",
            category_totals,
            [column for column in category_totals.columns if column != "Modelo"],
            fixed_columns=["Modelo"],
            columns_per_block=5,
        )

    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Comparador de custo de manter carros", layout="wide")
    st.title("Comparador de custo de manter carros")
    st.caption("Compare manter, comprar ou assinar: combustivel/energia, assinatura, IPVA, seguro, manutencao e custo de oportunidade.")

    ipva_default, horizon, investment_return = add_general_assumptions()

    export_container = st.container()
    with export_container:
        st.subheader("Exportar relatorio")
        st.caption("O botao aparece aqui depois que os dados da analise forem calculados.")

    st.subheader("Raciocinio")
    st.write(
        "Na compra, o valor do carro e usado para calcular IPVA e medir capital adicional ou liberado. "
        "Na assinatura, o app soma taxa inicial, mensalidades e custos fora do contrato, considerando que o valor "
        "do carro atual poderia ser vendido e ficar rendendo. A decisao fica: o custo de uso compensa comprar ou assinar?"
    )

    current = car_form(0, ipva_default, horizon)
    model_count = st.number_input("Quantidade de modelos para comparar", min_value=1, max_value=20, value=2, step=1)
    cars = [current] + [car_form(i, ipva_default, horizon, current.valor_base) for i in range(1, int(model_count) + 1)]

    simulations = pd.concat([simulate_car(car, current.valor_base, investment_return) for car in cars], ignore_index=True)
    final = simulations[simulations["Ano"] == horizon].copy().sort_values("Custo acumulado comparavel")
    current_final = final[final["ID"] == current.id].iloc[0]
    final["Diferenca vs carro atual"] = final["Custo acumulado comparavel"] - current_final["Custo acumulado comparavel"]
    final["Custo mensal efetivo"] = final["Custo acumulado comparavel"] / (horizon * 12)
    final["Diferenca mensal vs atual"] = final["Diferenca vs carro atual"] / (horizon * 12)
    winner = final.iloc[0]

    st.subheader("Resposta")
    c1, c2, c3 = st.columns(3)
    c1.metric("Melhor opcao", winner["Modelo"], money(winner["Custo acumulado comparavel"]))
    c2.metric("Mensal efetivo do melhor", money(winner["Custo mensal efetivo"]))
    cheaper_count = int((final["Diferenca vs carro atual"] < 0).sum())
    c3.metric("Modelos mais baratos que o atual", cheaper_count)

    if winner["ID"] == current.id:
        st.success("Pelas premissas informadas, compensa ficar com o carro atual.")
    else:
        st.success(f"Pelas premissas informadas, compensa trocar para {winner['Modelo']}.")

    with st.expander("Como o app chegou nessa resposta", expanded=True):
        st.markdown(
            f"""
1. Soma os custos anuais de uso de cada alternativa: combustivel/energia, assinatura, IPVA, seguro, licenciamento, revisoes, manutencao, pneus e outros.
2. Na compra, calcula a diferenca de capital contra o carro atual: valor do modelo menos valor do carro atual.
3. Na assinatura, considera que o valor do carro atual seria liberado e poderia render.
4. Aplica {investment_return:.1f}% ao ano sobre essa diferenca. Capital adicional vira custo; capital liberado vira beneficio.
5. Se houver financiamento do capital adicional, soma apenas os juros estimados, nao o valor principal do carro.
6. Vence quem tiver o menor custo acumulado comparavel no horizonte de {horizon} anos.
            """
        )

    st.subheader("Comparacao de cada modelo contra o atual")
    comparison_vs_current = final[final["ID"] != current.id][
        ["Modelo", "Diferenca vs carro atual", "Diferenca mensal vs atual", "Custo mensal efetivo", "Custo acumulado comparavel"]
    ].copy()
    comparison_vs_current["Leitura"] = np.where(
        comparison_vs_current["Diferenca vs carro atual"] < 0,
        "Economiza vs atual",
        np.where(comparison_vs_current["Diferenca vs carro atual"] > 0, "Custa mais que atual", "Empata com atual"),
    )
    comparison_vs_current["Economia vs atual"] = -comparison_vs_current["Diferenca vs carro atual"]
    st.dataframe(
        comparison_vs_current[
            [
                "Modelo",
                "Leitura",
                "Diferenca vs carro atual",
                "Diferenca mensal vs atual",
                "Economia vs atual",
                "Custo mensal efetivo",
                "Custo acumulado comparavel",
            ]
        ].style.format(
            {
                "Diferenca vs carro atual": money,
                "Diferenca mensal vs atual": money,
                "Economia vs atual": money,
                "Custo mensal efetivo": money,
                "Custo acumulado comparavel": money,
            }
        ),
        use_container_width=True,
    )

    cost_columns = [
        "Combustivel/energia",
        "IPVA",
        "Licenciamento",
        "Seguro",
        "Revisoes",
        "Manutencao",
        "Pneus",
        "Estacionamento",
        "Outros",
        "Assinatura",
        "Custo de oportunidade",
        "Juros do financiamento",
    ]
    breakdown = simulations.groupby("Modelo")[cost_columns].sum().reset_index()
    summary_base = final[
        [
            "Modelo",
            "Valor base IPVA",
            "Diferenca de capital vs atual",
            "Diferenca vs carro atual",
            "Custo mensal efetivo",
            "Custo acumulado comparavel",
        ]
    ]
    detailed_summary = summary_base.merge(breakdown, on="Modelo", how="left")
    ordered_cols = [
        "Modelo",
        "Valor base IPVA",
        "Diferenca de capital vs atual",
        "Diferenca vs carro atual",
        "Custo mensal efetivo",
        "Combustivel/energia",
        "IPVA",
        "Licenciamento",
        "Seguro",
        "Revisoes",
        "Manutencao",
        "Pneus",
        "Estacionamento",
        "Outros",
        "Assinatura",
        "Custo de oportunidade",
        "Juros do financiamento",
        "Custo acumulado comparavel",
    ]
    detailed_summary = detailed_summary[ordered_cols]

    with export_container:
        try:
            pdf_bytes = build_pdf_report(
                cars=cars,
                current=current,
                winner=winner,
                final=final,
                comparison_vs_current=comparison_vs_current[
                    [
                        "Modelo",
                        "Leitura",
                        "Diferenca vs carro atual",
                        "Diferenca mensal vs atual",
                        "Economia vs atual",
                        "Custo mensal efetivo",
                        "Custo acumulado comparavel",
                    ]
                ],
                detailed_summary=detailed_summary,
                simulations=simulations,
                breakdown=breakdown,
                investment_return=investment_return,
                horizon=horizon,
            )
            st.download_button(
                "Exportar relatorio em PDF",
                data=pdf_bytes,
                file_name="relatorio_comparacao_carros.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ModuleNotFoundError:
            st.warning("Para exportar em PDF, instale a dependencia matplotlib. O arquivo run_app.bat ja tenta instalar pelo JFrog.")

    st.dataframe(
        detailed_summary.style.format({col: money for col in ordered_cols if col != "Modelo"}),
        use_container_width=True,
    )

    line_data = simulations.copy()
    line_data["Rotulo"] = line_data["Custo acumulado comparavel"].map(short_money)
    line_fig = px.line(
        line_data,
        x="Ano",
        y="Custo acumulado comparavel",
        color="Modelo",
        markers=True,
        text="Rotulo",
        title="Custo acumulado para manter/usar cada opcao",
    )
    line_fig.update_traces(textposition="top center")
    st.plotly_chart(line_fig, use_container_width=True)

    monthly_fig = px.bar(
        final,
        x="Modelo",
        y="Custo mensal efetivo",
        color="Modelo",
        text=final["Custo mensal efetivo"].map(monthly_money),
        title="Custo mensal efetivo por modelo",
    )
    monthly_fig.update_traces(textposition="outside")
    st.plotly_chart(monthly_fig, use_container_width=True)

    bar_data = breakdown.melt(id_vars="Modelo", var_name="Categoria", value_name="Custo")
    bar_data["Rotulo"] = bar_data.apply(lambda row: cost_label(row["Categoria"], row["Custo"]), axis=1)
    bar_fig = px.bar(
        bar_data,
        x="Modelo",
        y="Custo",
        color="Categoria",
        text="Rotulo",
        title="Composicao do custo no horizonte",
    )
    bar_fig.update_traces(textposition="inside", insidetextanchor="middle")
    totals = breakdown.set_index("Modelo")[cost_columns].sum(axis=1).reset_index(name="Total")
    bar_fig.add_trace(
        go.Scatter(
            x=totals["Modelo"],
            y=totals["Total"],
            text=totals["Total"].map(short_money),
            mode="text",
            textposition="top center",
            showlegend=False,
            textfont=dict(size=13, color="black"),
            name="Total",
        )
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Curva de ponto de virada")
    target_options = {f"{car.nome} ({car.id})": car for car in cars if car.id != current.id}
    target_label = st.selectbox("Modelo para testar contra o carro atual", list(target_options))
    target = target_options[target_label]
    surface = make_break_even(current, target, investment_return)
    unit = "km/kWh" if target.tipo == "Eletrico" else "km/l"
    fig = go.Figure(
        data=go.Contour(
            x=surface["Valor do carro"],
            y=surface["Consumo"],
            z=surface["Diferenca vs atual"],
            colorscale="RdYlGn_r",
            contours=dict(showlabels=True),
            colorbar=dict(title="R$ vs atual"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target.valor_base],
            y=[target.consumo_km_kwh if target.tipo == "Eletrico" else target.consumo_km_l],
            mode="markers",
            marker=dict(size=12, color="black"),
            name="Premissa atual",
        )
    )
    fig.update_layout(
        title=f"Preco e consumo que fariam {target.nome} bater {current.nome}",
        xaxis_title="Valor do carro",
        yaxis_title=f"Consumo ({unit})",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Valores negativos indicam que o modelo testado fica mais barato que manter o carro atual.")


if __name__ == "__main__":
    main()
