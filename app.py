# Title: Compare Car
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal
import json
import os

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


SIMULATIONS_DIR = Path("simulacoes")
SIMULATIONS_DIR.mkdir(exist_ok=True)


def save_simulation(name: str, ipva_default: float, horizon: int, investment_return: float, cars: list[CarInputs]) -> None:
    data = {
        "ipva_default": ipva_default,
        "horizon": horizon,
        "investment_return": investment_return,
        "cars": [car.__dict__ for car in cars],
    }
    file_path = SIMULATIONS_DIR / f"{name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_simulation(name: str) -> tuple[float, int, float, list[CarInputs]]:
    file_path = SIMULATIONS_DIR / f"{name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Simulation file '{name}.json' not found.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cars = [CarInputs(**car_data) for car_data in data["cars"]]
        return data["ipva_default"], data["horizon"], data["investment_return"], cars
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{name}.json': {e}")
    except KeyError as e:
        raise ValueError(f"Missing expected key in '{name}.json': {e}")


def list_simulations() -> list[str]:
    return [f.stem for f in SIMULATIONS_DIR.glob("*.json")]


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
                    ("Parcela mensal", monthly_money(financing_monthly_payment(car, capital_diff))),
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


def financing_principal(car: CarInputs, capital_difference: float) -> float:
    if car.carro_atual or car.pagamento != "Financiado" or capital_difference <= 0:
        return 0.0
    return max(capital_difference - car.entrada_extra, 0.0)


def financing_monthly_payment(car: CarInputs, capital_difference: float) -> float:
    return monthly_payment(financing_principal(car, capital_difference), car.taxa_juros_am / 100, car.prazo_meses)


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

def financing_schedule_by_year(car: CarInputs, capital_difference: float) -> list[dict[str, float]]:
    schedule = []
    principal = financing_principal(car, capital_difference)
    monthly_rate = car.taxa_juros_am / 100
    monthly_installment = financing_monthly_payment(car, capital_difference)
    balance = principal
    cumulative_amortization = 0.0

    for year in range(1, car.horizonte_anos + 1):
        annual_payment = 0.0
        annual_interest = 0.0
        annual_amortization = 0.0

        for month in range(1, 13):
            absolute_month = (year - 1) * 12 + month
            if absolute_month > car.prazo_meses or balance <= 0:
                break

            interest = balance * monthly_rate
            amortization = min(monthly_installment - interest, balance)
            payment = interest + amortization
            balance = max(balance - amortization, 0.0)

            annual_payment += payment
            annual_interest += interest
            annual_amortization += amortization

        cumulative_amortization += annual_amortization
        schedule.append(
            {
                "Parcela mensal financiamento": monthly_installment if annual_payment > 0 else 0.0,
                "Parcelas financiamento": annual_payment,
                "Juros financiamento": annual_interest,
                "Amortizacao financiamento": annual_amortization,
                "Valor quitado financiamento": car.entrada_extra + cumulative_amortization,
                "Saldo financiamento": balance,
            }
        )

    return schedule


def simulate_car(car: CarInputs, current_value: float, investment_return: float) -> pd.DataFrame:

    if car.carro_atual:
        capital_difference = 0.0

    elif car.pagamento == "Assinatura":
        capital_difference = -current_value

    else:
        capital_difference = car.valor_base - current_value

    financing_schedule = financing_schedule_by_year(car, capital_difference)

    rows = []
    cumulative = 0.0

    for year in range(1, car.horizonte_anos + 1):

        revision = (
            car.revisoes_anuais[year - 1]
            if year <= len(car.revisoes_anuais)
            else 0
        )

        subscription = (
            car.assinatura_mensal * 12
            + (
                car.taxa_inicial_assinatura
                if year == 1
                else 0
            )
            if car.pagamento == "Assinatura"
            else 0.0
        )

        ipva = (
            0.0
            if car.pagamento == "Assinatura"
            else car.valor_base * car.ipva_percentual / 100
        )

        licenciamento = (
            0.0
            if car.pagamento == "Assinatura"
            else car.licenciamento_anual
        )

        seguro = (
            0.0
            if car.pagamento == "Assinatura"
            else car.seguro_anual
        )

        revision = (
            0.0
            if car.pagamento == "Assinatura"
            else revision
        )

        manutencao = (
            0.0
            if car.pagamento == "Assinatura"
            else car.manutencao_anual
        )

        pneus = (
            0.0
            if car.pagamento == "Assinatura"
            else car.pneus_anual
        )

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

        financing_year = financing_schedule[year - 1]
        if car.pagamento == "Financiado":
            # Custo de oportunidade sobre o capital líquido desembolsado:
            # entrada paga agora menos valor recebido da venda do carro atual
            invested_capital = max(0, car.entrada_extra - current_value)
        else:
            invested_capital = capital_difference
        annual_opportunity = invested_capital * investment_return / 100

        total = operating + annual_opportunity + financing_year["Juros financiamento"]

        cumulative += total

        rows.append(
            {
                "Modelo": car.nome,
                "ID": car.id,
                "Ano": year,
                "Valor base IPVA": car.valor_base,
                "Diferenca de capital vs atual": capital_difference,
                "Capital imobilizado": invested_capital,
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
                "Valor financiado": financing_principal(car, capital_difference),
                "Entrada financiamento": car.entrada_extra if car.pagamento == "Financiado" and year == 1 else 0.0,
                "Parcela mensal financiamento": financing_year["Parcela mensal financiamento"],
                "Parcelas financiamento": financing_year["Parcelas financiamento"],
                "Juros financiamento": financing_year["Juros financiamento"],
                "Amortizacao financiamento": financing_year["Amortizacao financiamento"],
                "Valor quitado financiamento": financing_year["Valor quitado financiamento"],
                "Saldo financiamento": financing_year["Saldo financiamento"],
                "Custo operacional": operating,
                "Custo anual comparavel": total,
                "Custo acumulado comparavel": cumulative,
            }
        )

    return pd.DataFrame(rows)


def add_general_assumptions(ipva_default_val: float = 4.0, horizon_val: int = 5, investment_return_val: float = 10.0) -> tuple[float, int, float]:
    st.subheader("📋 Premissas Gerais da Análise")
    col1, col2, col3 = st.columns(3)

    state = col1.selectbox("🏠 Estado", sorted(STATE_IPVA_DEFAULTS), index=sorted(STATE_IPVA_DEFAULTS).index("SP"), help="Selecione o estado onde o veículo será registrado para aplicar a alíquota de IPVA correta")
    ipva_default = col1.number_input(
        "📊 Alíquota Padrão de IPVA (% ao ano)",
        min_value=0.0,
        max_value=10.0,
        value=ipva_default_val,
        step=0.1,
        help="Percentual padrão do estado para IPVA. Nota: a alíquota real pode variar conforme combustível, ano de fabricação e possíveis isenções. Você pode ajustar por carro específico.",
    )
    horizon = col2.slider("⏱️ Período de Análise (anos)", 1, 10, horizon_val, help="Quantos anos você pretende manter ou usar o veículo? Quanto maior, mais custos de manutenção acumulam.")
    investment_return = col3.number_input(
        "💰 Retorno Esperado do Investimento (% ao ano)",
        min_value=0.0,
        max_value=30.0,
        value=investment_return_val,
        step=0.5,
        help="Qual seria o retorno do dinheiro se você não o gastasse na troca? (ex: aplicação financeira, CDI, poupança). Usado para calcular o 'custo de oportunidade' - o valor que você deixaria de ganhar investindo esse dinheiro.",
    )
    return ipva_default, horizon, investment_return


def car_form(index: int, ipva_default: float, horizon: int, current_value: float | None = None, default_car: CarInputs | None = None) -> CarInputs:
    current = index == 0
    label = "Carro atual" if current else f"Modelo {index}"

    # Defaults
    default_nome = default_car.nome if default_car else label
    default_tipo = default_car.tipo if default_car else "Combustao"
    default_pagamento = default_car.pagamento if default_car and not current else "A vista"
    default_valor_base = default_car.valor_base if default_car else (80_000.0 if current else 120_000.0)
    default_km_mes = default_car.km_mes if default_car else 1200.0
    default_ipva = default_car.ipva_percentual if default_car else ipva_default
    default_seguro = default_car.seguro_anual if default_car else (default_valor_base * 0.035)
    default_licenciamento = default_car.licenciamento_anual if default_car else 250.0
    default_manutencao = default_car.manutencao_anual if default_car else 1000.0
    default_pneus = default_car.pneus_anual if default_car else 800.0
    default_estacionamento = default_car.estacionamento_anual if default_car else 0.0
    default_outros = default_car.outros_anuais if default_car else 0.0
    default_revisoes = list(default_car.revisoes_anuais) if default_car else [1200.0 + 250.0 * ((year - 1) % 3) for year in range(1, horizon + 1)]
    default_consumo_l = default_car.consumo_km_l if default_car else (11.0 if default_tipo == "Combustao" else 16.0)
    default_consumo_kwh = default_car.consumo_km_kwh if default_car else 6.0
    default_preco_combustivel = default_car.preco_combustivel if default_car else 6.00
    default_preco_kwh = default_car.preco_kwh if default_car else 0.95
    default_percentual_recarga = default_car.percentual_recarga_externa if default_car else 0.0
    default_entrada_extra = default_car.entrada_extra if default_car else 0.0
    default_taxa = default_car.taxa_juros_am if default_car else 1.3
    default_prazo = default_car.prazo_meses if default_car else 48
    default_assinatura_mensal = default_car.assinatura_mensal if default_car else 3500.0
    default_taxa_inicial_assinatura = default_car.taxa_inicial_assinatura if default_car else 0.0

    with st.expander(label, expanded=index <= 2):
        col1, col2, col3 = st.columns(3)

        with col1:
            nome = st.text_input("🚗 Nome/Modelo", value=default_nome, key=f"name_{index}", help="Identifique o veículo (ex: 'HB20 1.0', 'Tiggo 8', 'e-Golf 2025')")
            pagamento = default_pagamento
            if not current:
                pagamento = st.selectbox("💳 Forma de Pagamento", ["A vista", "Financiado", "Assinatura"], index=["A vista", "Financiado", "Assinatura"].index(default_pagamento), key=f"pay_{index}", help="Como o veículo será adquirido: compra à vista, compra financiada, ou contrato de assinatura/aluguel")
            tipo = st.selectbox("⚡ Tipo de Combustível/Energia", ["Combustao", "Hibrido", "Eletrico"], index=["Combustao", "Hibrido", "Eletrico"].index(default_tipo), key=f"type_{index}", help="Escolha o tipo de propulsão do veículo")
            valor_default = 80_000.0 if current else 120_000.0
            if current:
                valor_label = "💵 Valor de Mercado Atual"
                help_text = "Valor do seu veículo atual (para venda). Usado para calcular o capital liberado ou adicional necessário na troca."
            elif pagamento == "Assinatura":
                valor_label = "💵 Valor de Referência do Modelo"
                help_text = "Valor de mercado aproximado do modelo. Serve apenas como referência para calcular o seguro proporcional; não afeta o custo total da assinatura."
            else:
                valor_label = "💵 Preço do Modelo (à vista)"
                help_text = "Preço de mercado do veículo novo. Será usado para calcular IPVA e o capital adicional/liberado em relação ao carro atual."
            
            valor_base = st.number_input(
                valor_label,
                0.0,
                2_000_000.0,
                default_valor_base,
                1_000.0,
                key=f"value_{index}",
                help=help_text,
            )
            km_mes = st.number_input("📍 Quilometragem Mensal (km)", 0.0, 20_000.0, default_km_mes, 100.0, key=f"km_{index}", help="Quantos km você estima rodar por mês? Afeta o custo de combustível/energia, pneus e manutenção.")
            ipva = default_ipva
            if pagamento != "Assinatura":
                ipva = st.number_input("📋 IPVA (% ao ano)", 0.0, 10.0, default_ipva, 0.1, key=f"ipva_{index}", help="Alíquota do Imposto sobre Propriedade de Veículos Automotores. Pode variar conforme o tipo de combustível e ano do veículo no seu estado.")
            if not current and current_value is not None:
                if pagamento == "Assinatura":
                    st.info(f"💰 **Capital Liberado ao Vender o Carro Atual:** {money(current_value)}\n\nEsse valor poderá ser investido, gerando rendimentos que reduzem o custo efetivo da assinatura.")
                else:
                    diff = valor_base - current_value
                    if diff >= 0:
                        st.warning(f"💸 **Capital Adicional Necessário:** {money(diff)}\n\nVocê precisará de {money(diff)} além do valor do carro atual para fazer essa troca (à vista ou financiar essa diferença).")
                    else:
                        st.success(f"💰 **Capital Liberado vs Carro Atual:** {money(abs(diff))}\n\nO novo modelo é mais barato! Você libera {money(abs(diff))} que poderá ser investido.")

        with col2:
            consumo_l = default_consumo_l
            consumo_kwh = default_consumo_kwh
            preco_combustivel = default_preco_combustivel
            preco_kwh = default_preco_kwh
            percentual_recarga_externa = default_percentual_recarga

            if tipo == "Combustao":
                consumo_l = st.number_input("⛽ Consumo de Combustível (km/l)", 0.1, 100.0, default_consumo_l, 0.5, key=f"km_l_{index}", help="Quantos quilômetros o carro percorre com 1 litro de combustível. Dados do fabricante ou do seu uso real.")
                preco_combustivel = st.number_input("💰 Preço da Gasolina (R$/litro)", 0.0, 20.0, default_preco_combustivel, 0.05, key=f"fuel_price_{index}", help="Preço atual ou estimado do combustível na sua região.")

            if tipo == "Hibrido":
                consumo_l = st.number_input(
                    "⛽ Consumo no Modo Motor (km/l equivalente)",
                    0.1,
                    100.0,
                    default_consumo_l,
                    0.5,
                    key=f"km_l_{index}",
                    help="Consumo quando a bateria está descarregada e o motor a combustão sustenta a energia. Use dados realistas da sua experiência.",
                )
                preco_combustivel = st.number_input("💰 Preço da Gasolina (R$/litro)", 0.0, 20.0, default_preco_combustivel, 0.05, key=f"fuel_price_{index}", help="Preço atual ou estimado do combustível na sua região.")
                percentual_recarga_externa = st.slider(
                    "🔌 Percentual de Energia Carregada na Tomada (%)",
                    0.0,
                    100.0,
                    default_percentual_recarga,
                    5.0,
                    key=f"external_charge_{index}",
                    help="Que porcentagem da energia vem de carregamento externo (tomada)? Ex: 50% = metade do trajeto com energia elétrica.",
                )

            if tipo == "Eletrico" or (tipo == "Hibrido" and percentual_recarga_externa > 0):
                consumo_kwh = st.number_input("⚡ Eficiência Energética (km/kWh)", 0.1, 20.0, default_consumo_kwh, 0.1, key=f"km_kwh_{index}", help="Quantos quilômetros o carro percorre com 1 kWh de energia. Valores típicos: 4-7 km/kWh.")
                preco_kwh = st.number_input("💰 Tarifa de Energia (R$/kWh)", 0.0, 5.0, default_preco_kwh, 0.05, key=f"kwh_price_{index}", help="Preço da energia elétrica na sua região (residencial ou pública).")

        with col3:
            entrada_extra = 0.0
            taxa = default_taxa
            prazo = default_prazo
            assinatura_mensal = default_assinatura_mensal
            taxa_inicial_assinatura = default_taxa_inicial_assinatura

            if pagamento == "Financiado":
                capital_diff = valor_base - current_value if not current and current_value is not None else 0
                entrada_percentual = (default_entrada_extra / valor_base * 100) if valor_base > 0 else 20.0
                entrada_percentual = st.slider("💵 Percentual de Entrada (% do valor total)", 0.0, 100.0, entrada_percentual, 5.0, key=f"down_pct_{index}", help="Quanto você vai dar de entrada? 20-30% é comum. Quanto maior, menores os juros totais.")
                entrada_extra = valor_base * entrada_percentual / 100
                taxa = st.number_input("📈 Taxa de Juros Mensais (%)", 0.0, 10.0, default_taxa, 0.05, key=f"rate_{index}", help="Taxa mensal do financiamento. Típica: 0,8% a 2,5% a.m. (9,6% a 30% ao ano).")
                prazo = st.number_input("⏱️ Prazo do Financiamento (meses)", 1, 120, default_prazo, 6, key=f"term_{index}", help="Quantos meses para pagar? Comum: 24, 36, 48 ou 60 meses.")
                if entrada_extra > 0:
                    st.info(f"✅ **Entrada Calculada:** {money(entrada_extra)}\n\nVoc\u00ea pagará esse valor hoje. O restante será parcelado nos {int(prazo)} meses seguintes.")
                # Calcular e exibir parcela mensal
                if capital_diff > 0 and entrada_extra < capital_diff:
                    parcela_mensal = financing_monthly_payment(
                        CarInputs(
                            id="",
                            nome="",
                            carro_atual=False,
                            tipo="Combustao",
                            valor_base=valor_base,
                            pagamento="Financiado",
                            entrada_extra=entrada_extra,
                            taxa_juros_am=taxa,
                            prazo_meses=int(prazo),
                            assinatura_mensal=0,
                            taxa_inicial_assinatura=0,
                            km_mes=0,
                            consumo_km_l=0,
                            consumo_km_kwh=0,
                            preco_combustivel=0,
                            preco_kwh=0,
                            percentual_recarga_externa=0,
                            ipva_percentual=0,
                            licenciamento_anual=0,
                            seguro_anual=0,
                            revisoes_anuais=(),
                            manutencao_anual=0,
                            pneus_anual=0,
                            estacionamento_anual=0,
                            outros_anuais=0,
                            horizonte_anos=horizon,
                        ),
                        capital_diff
                    )
                    st.success(f"💳 **Parcela Mensal Estimada:** {monthly_money(parcela_mensal)}\n\nEsse é o valor que você pagará por mês durante o financiamento (apenas principal + juros, sem outros custos).")

            if pagamento == "Assinatura":
                assinatura_mensal = st.number_input("📅 Mensalidade da Assinatura (R$/mês)", 0.0, 100_000.0, default_assinatura_mensal, 100.0, key=f"sub_month_{index}", help="Valor mensal que você pagará. Geralmente inclui seguro, manutenção e assistência.")
                taxa_inicial_assinatura = st.number_input("💳 Taxa Inicial/Adesão (R$)", 0.0, 100_000.0, default_taxa_inicial_assinatura, 500.0, key=f"sub_fee_{index}", help="Custo único no início do contrato de assinatura.")
                seguro = 0.0
                licenciamento = 0.0
                st.info("✅ Na assinatura, IPVA, seguro, licenciamento, revisões, manutenção e pneus estão geralmente inclusos. Abaixo informamos apenas custos adicionais (estacionamento, etc.).")
            else:
                seguro = st.number_input("🛡️ Seguro Anual (R$)", 0.0, 100_000.0, default_seguro, 250.0, key=f"ins_{index}", help="Prêmio anual do seguro do veículo. Geralmente é ~3-4% do valor do carro.")
                licenciamento = st.number_input("📝 Licenciamento Anual (R$)", 0.0, 10_000.0, default_licenciamento, 50.0, key=f"lic_{index}", help="Valor anual do licenciamento/emplacamento. Típico: R$ 200-400.")

        if pagamento == "Assinatura":
            st.caption("🚗 Custos Adicionais (não inclusos na assinatura)")
            c1, c2 = st.columns(2)
            manutencao = 0.0
            pneus = 0.0
            estacionamento = c1.number_input("🅿️ Estacionamento/ano (R$)", 0.0, 100_000.0, default_estacionamento, 100.0, key=f"park_{index}", help="Custo anual de estacionamento, garagem, valet, etc.")
            outros = c2.number_input("📦 Outros Custos/ano (R$)", 0.0, 100_000.0, default_outros, 100.0, key=f"other_{index}", help="Multas, pedágios, lavagens frequentes, acessórios, etc.")
            revisoes = tuple(0.0 for _ in range(horizon))
        else:
            st.caption("🔧 Custos de Manutenção e Uso do Veículo")
            c1, c2, c3, c4 = st.columns(4)
            manutencao = c1.number_input("🔧 Manutenção/ano (R$)", 0.0, 100_000.0, default_manutencao, 100.0, key=f"maint_{index}", help="Óleos, filtros, fluidos, correias, etc. Típico: R$ 500-1500/ano.")
            pneus = c2.number_input("⛩️ Pneus/ano (R$)", 0.0, 100_000.0, default_pneus, 100.0, key=f"tires_{index}", help="Rodízio, reposição e balanceamento. Típico: R$ 800-1500/ano.")
            estacionamento = c3.number_input("🅿️ Estacionamento/ano (R$)", 0.0, 100_000.0, default_estacionamento, 100.0, key=f"park_{index}", help="Custo anual de estacionamento, garagem, valet, etc.")
            outros = c4.number_input("📦 Outros/ano (R$)", 0.0, 100_000.0, default_outros, 100.0, key=f"other_{index}", help="Multas, pedágios, lavagens, acessórios, etc.")

            st.caption("💰 Custos de Revisão por Ano")
            revision_columns = st.columns(min(horizon, 5))
            revisoes = []
            for year in range(1, horizon + 1):
                column = revision_columns[(year - 1) % len(revision_columns)]
                revisoes.append(
                    column.number_input(
                        f"Ano {year}",
                        0.0,
                        100_000.0,
                        default_revisoes[year - 1] if year <= len(default_revisoes) else 1200.0 + 250.0 * ((year - 1) % 3),
                        100.0,
                        key=f"rev_{index}_{year}",
                        help=f"Custo estimado de revisão no ano {year}. Revisões maiores a cada 3 anos (revisão de 40mil km).",
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

        # PDF graph for financed cars paid off amount
        financed_sim = simulations[simulations["Valor financiado"] > 0]
        if not financed_sim.empty:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            for model, rows in financed_sim.groupby("Modelo"):
                ax.plot(rows["Ano"], rows["Valor quitado financiamento"], marker="o", label=model)
                for _, row in rows.iterrows():
                    ax.annotate(
                        short_money(row["Valor quitado financiamento"]),
                        (row["Ano"], row["Valor quitado financiamento"]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=7,
                    )
            ax.set_title("Montante quitado do carro novo (financiamento)", fontweight="bold")
            ax.set_xlabel("Ano")
            ax.set_ylabel("Valor quitado")
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
    st.title("🚗 Comparador de Custo de Manter Veículos")
    st.caption("📃 Compare o custo total: comprar à vista, financiar ou assinar. Lev em conta combustível/energia, assinatura, IPVA, seguro, manutenção, revisões e custo de oportunidade do capital.")

    # Carregar valores da sessão se existirem
    ipva_default_val = st.session_state.get("ipva_default", 4.0)
    horizon_val = st.session_state.get("horizon", 5)
    investment_return_val = st.session_state.get("investment_return", 10.0)
    cars_loaded = st.session_state.get("cars", [])

    ipva_default, horizon, investment_return = add_general_assumptions(ipva_default_val, horizon_val, investment_return_val)

    # Carregar Simulação (antes dos forms)
    with st.expander("📂 Carregar Simulação Salva", expanded=False):
        simulations = list_simulations()
        if simulations:
            selected_sim = st.selectbox("Selecione uma simulação para carregar", [""] + simulations, key="load_sim", help="Clique para restaurar uma simulação anterior com todos os seus parâmetros.")
            if st.button("✅ Carregar", key="btn_load", use_container_width=True) and selected_sim and selected_sim != "":
                try:
                    ipva_default, horizon, investment_return, cars_loaded_temp = load_simulation(selected_sim)
                    st.session_state["ipva_default"] = ipva_default
                    st.session_state["horizon"] = horizon
                    st.session_state["investment_return"] = investment_return
                    st.session_state["cars"] = cars_loaded_temp
                    st.success(f"✅ Simulação '{selected_sim}' carregada! Recarregue a página para ver as mudanças (pressione **F5**).")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
        else:
            st.info("ℹ️ Nenhuma simulação salva ainda. Crie uma novo análise e salve-a!")

    st.subheader("🧠 Como Funciona a Análise")
    st.write(
        "**Na compra:** O valor do carro é usado para calcular IPVA e medir quanto de capital adicional (ou liberado) está envolvido na troca. "
        "**Na assinatura:** Somamos taxa inicial, mensalidades e custos adicionais, reconhecendo que o dinheiro do carro atual poderia ser investido. "
        "**A pergunta principal:** Qual opção (manter, comprar ou assinar) gera o menor custo total considerando todos esses fatores ao longo do período analisado?"
    )

    current = car_form(0, ipva_default, horizon, default_car=cars_loaded[0] if cars_loaded else None)
    model_count = st.number_input("💶 Quantos Modelos Alternativos Comparar?", min_value=1, max_value=20, value=len(cars_loaded) - 1 if cars_loaded else 2, step=1, help="Quantos outros carros/opções você quer comparar contra o atual?")  
    cars = [current] + [car_form(i, ipva_default, horizon, current.valor_base, default_car=cars_loaded[i] if cars_loaded and i < len(cars_loaded) else None) for i in range(1, int(model_count) + 1)]

    # Salvar e Gerenciar Simulações (depois dos forms)
    with st.expander("💾 Salvar e Gerenciar Simulações", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💿 Salvar Simulação Atual")
            save_name = st.text_input("Nome da simulação", key="save_name", placeholder="Ex: HB20 vs Gol - Mar/2026")
            if st.button("💾 Salvar", key="btn_save", use_container_width=True) and save_name:
                try:
                    st.session_state["ipva_default"] = ipva_default
                    st.session_state["horizon"] = horizon
                    st.session_state["investment_return"] = investment_return
                    st.session_state["cars"] = cars
                    save_simulation(save_name, ipva_default, horizon, investment_return, cars)
                    st.success(f"✅ Simulação '{save_name}' salva com sucesso!")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {e}")
        
        with col2:
            st.subheader("👁️ Visualizar/Excluir Simulação")
            simulations = list_simulations()
            if simulations:
                manage_sim = st.selectbox("Selecione uma simulação para gerenciar", [""] + simulations, key="manage_sim")
                if manage_sim and manage_sim != "":
                    if st.button("📋 Mostrar Preview", key="btn_preview", use_container_width=True):
                        try:
                            prev_ipva, prev_horizon, prev_return, prev_cars = load_simulation(manage_sim)
                            st.write(f"**📋 Nome:** {manage_sim}")
                            st.write(f"**🏷️ IPVA Padrão:** {prev_ipva}%")
                            st.write(f"**⏱️ Horizonte:** {prev_horizon} anos")
                            st.write(f"**💹 Rendimento Esperado:** {prev_return}% a.a.")
                            st.write(f"**🚗 Número de Carros:** {len(prev_cars)}")
                            for i, car in enumerate(prev_cars):
                                with st.expander(f"🚙 Carro {i+1}: {car.nome}"):
                                    st.write(f"**Tipo:** {car.tipo}")
                                    st.write(f"**Pagamento:** {car.pagamento}")
                                    st.write(f"**Valor:** R$ {car.valor_base:,.0f}")
                                    if car.pagamento == "Financiado":
                                        st.write(f"**Entrada:** R$ {car.entrada_extra:,.0f}")
                                        st.write(f"**Juros:** {car.taxa_juros_am}% a.m.")
                                        st.write(f"**Prazo:** {car.prazo_meses} meses")
                                    elif car.pagamento == "Assinatura":
                                        st.write(f"**Mensalidade:** R$ {car.assinatura_mensal:,.0f}")
                                        st.write(f"**Taxa Inicial:** R$ {car.taxa_inicial_assinatura:,.0f}")
                        except Exception as e:
                            st.error(f"❌ Erro ao carregar preview: {e}")
                    
                    if st.button("🗑️ Excluir Simulação", key="btn_delete", use_container_width=True):
                        try:
                            file_path = SIMULATIONS_DIR / f"{manage_sim}.json"
                            if file_path.exists():
                                file_path.unlink()
                                st.success(f"✅ Simulação '{manage_sim}' excluída!")
                                st.rerun()
                            else:
                                st.error("❌ Arquivo não encontrado.")
                        except Exception as e:
                            st.error(f"❌ Erro ao excluir: {e}")
            else:
                st.info("ℹ️ Nenhuma simulação para gerenciar.")

    simulations = pd.concat([simulate_car(car, current.valor_base, investment_return) for car in cars], ignore_index=True)
    final = simulations[simulations["Ano"] == horizon].copy().sort_values("Custo acumulado comparavel")
    current_final = final[final["ID"] == current.id].iloc[0]
    final["Diferenca vs carro atual"] = final["Custo acumulado comparavel"] - current_final["Custo acumulado comparavel"]
    final["Custo mensal efetivo"] = final["Custo acumulado comparavel"] / (horizon * 12)
    final["Diferenca mensal vs atual"] = final["Diferenca vs carro atual"] / (horizon * 12)
    winner = final.iloc[0]

    st.subheader("🎯 Resultado da Análise")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Melhor Opção", winner["Modelo"], money(winner["Custo acumulado comparavel"]), help="Opção com menor custo total acumulado no período analisado")
    c2.metric("💰 Custo Mensal Efetivo", money(winner["Custo mensal efetivo"]), help="Custo total dividido pelos meses do período analisado")
    cheaper_count = int((final["Diferenca vs carro atual"] < 0).sum())
    c3.metric("📊 Opções Mais Econômicas", cheaper_count, help="Quantos modelos saem mais baratos que manter o carro atual")

    if winner["ID"] == current.id:
        st.success("✅ **Pela análise realizada, É MAIS ECONÔMICO FICAR COM O CARRO ATUAL.**\n\nOs custos de manutenção, depreciação (se aplicavel) e capital associados aos modelos alternativos superam os beneficios. Considere manter seu veículo atual neste período.")
    else:
        st.success(f"✅ **Pela análise realizada, É MAIS ECONÔMICO TROCAR PARA {winner['Modelo'].upper()}.**\n\nO modelo sugerido gera um custo total {money(abs(winner['Diferenca vs carro atual']))} {'maior' if winner['Diferenca vs carro atual'] > 0 else 'menor'} que manter o carro atual.")

    with st.expander("📖 Como o App Chegou Nessa Resposta?", expanded=True):
        st.markdown(
            f"""
1. **Custos Operacionais:** Soma todos os custos anuais de uso: combustível/energia, assinatura, IPVA, seguro, licenciamento, revisões, manutenção, pneus e outros.

2. **Capital Inicial:** 
   - Na **compra**: Calcula a diferença de capital = (Preço do modelo novo) - (Valor do carro atual)
   - Na **assinatura**: Considera que o carro atual seria vendido e o dinheiro poderia ser investido

3. **Custo de Oportunidade:** Aplica **{investment_return:.1f}% ao ano** sobre o capital envolvido. 
   - Capital adicional = custo (você gasta mais dinheiro)
   - Capital liberado = benefício (você ganha investindo)

4. **Juros do Financiamento:** Se o carro for financiado, soma apenas os juros (não principal), pois o principal é o capital inicial.

5. **Vencedor:** A opção com **menor custo acumulado comparável no horizonte de {horizon} anos** vence.

**Interpretação:** Custos negativos vs atual = mais econômico. Custos positivos = mais caro.
            """
        )

    st.subheader("📊 Comparação Detalhada: Cada Modelo vs. Carro Atual")
    comparison_vs_current = final[final["ID"] != current.id][
        ["Modelo", "Diferenca vs carro atual", "Diferenca mensal vs atual", "Custo mensal efetivo", "Custo acumulado comparavel"]
    ].copy()
    comparison_vs_current["Leitura"] = np.where(
        comparison_vs_current["Diferenca vs carro atual"] < 0,
        "✅ Mais Econômico",
        np.where(comparison_vs_current["Diferenca vs carro atual"] > 0, "❌ Mais Caro", "➡️ Mesmo Custo"),
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
        "Parcelas financiamento",
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
        "Parcelas financiamento",
        "Custo acumulado comparavel",
    ]
    detailed_summary = detailed_summary[ordered_cols]

    st.subheader("📄 Exportar Relatório em PDF")
    st.caption("Relatório completo com premissas, gráficos e tabelas detalhadas da análise.")
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
            "📥 Exportar Relató rio em PDF",
            data=pdf_bytes,
            file_name="relatorio_comparacao_carros.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Gera um PDF completo com premissas, gráficos e tabelas da análise."
        )
    except ModuleNotFoundError:
        st.warning("⚠️ Para exportar em PDF, instale a dependência `matplotlib`. O arquivo `run_app.bat` já tenta instalar automaticamente via JFrog.")

    st.dataframe(
        detailed_summary.style.format({col: money for col in ordered_cols if col != "Modelo"}),
        use_container_width=True,
    )

    # Tabela resumida de custos por categoria
    st.subheader("💸 Resumo de Custos por Categoria (Total no Período Analisado)")
    st.caption("📋 Veja a decomposição de cada categoria de custo (combustível, IPVA, seguro, etc.) para cada modelo ao longo de todo o período.")
    summary_table = breakdown.copy()
    summary_table["Total custo"] = summary_table[cost_columns].sum(axis=1)
    summary_table = summary_table[["Modelo"] + cost_columns + ["Total custo"]]
    st.dataframe(
        summary_table.style.format({col: money for col in summary_table.columns if col != "Modelo"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("📈 Gráficos de Análise")
    st.markdown("Visualizações para entender melhor os custos e fazer comparações entre os modelos.")
    
    st.markdown("**1️⃣ Custo Acumulado ao Longo do Tempo** - Veja como o custo total cresceria a cada ano para cada opção")
    line_data = simulations.copy()
    line_data["Rotulo"] = line_data["Custo acumulado comparavel"].map(short_money)
    line_fig = px.line(
        line_data,
        x="Ano",
        y="Custo acumulado comparavel",
        color="Modelo",
        markers=True,
        text="Rotulo",
        title="📈 Custo Acumulado para Manter/Usar Cada Opção",
    )
    line_fig.update_traces(textposition="top center")

    # Adicionar linha para saldo de financiamento (parcelas restantes)
    financed_models = simulations[simulations["Valor financiado"] > 0]["Modelo"].unique()
    for model in financed_models:
        model_data = simulations[simulations["Modelo"] == model]
        saldo_data = model_data[model_data["Saldo financiamento"] > 0]
        if not saldo_data.empty:
            line_fig.add_trace(
                go.Scatter(
                    x=saldo_data["Ano"],
                    y=saldo_data["Saldo financiamento"],
                    mode="lines+markers+text",
                    name=f"{model} - Saldo financiamento",
                    line=dict(dash="dash", color="red"),
                    marker=dict(symbol="triangle-down"),
                    text=saldo_data["Saldo financiamento"].map(short_money),
                    textposition="bottom center",
                    showlegend=True,
                )
            )

    st.plotly_chart(line_fig, use_container_width=False, width=800, height=400)

    # Graph for financed cars paid off amount
    financed_data = simulations[simulations["Valor financiado"] > 0].copy()
    if not financed_data.empty:
        st.markdown("**2️⃣ Montante Quitado do Financiamento** - Quanto do carro já foi pago (financiamentos apenas)")
        financed_data["Rotulo_quitado"] = financed_data["Valor quitado financiamento"].map(short_money)
        financed_fig = px.line(
            financed_data,
            x="Ano",
            y="Valor quitado financiamento",
            color="Modelo",
            markers=True,
            text="Rotulo_quitado",
            title="📊 Montante Quitado do Carro Novo (Financiamento)",
        )
        financed_fig.update_traces(textposition="top center")
        st.plotly_chart(financed_fig, use_container_width=False, width=800, height=400)

    st.markdown("**3️⃣ Custo Mensal Efetivo** - Custo total dividido pelos meses (para comparação fácil)")
    monthly_fig = px.bar(
        final,
        x="Modelo",
        y="Custo mensal efetivo",
        color="Modelo",
        text=final["Custo mensal efetivo"].map(monthly_money),
        title="📊 Custo Mensal Efetivo por Modelo",
    )
    monthly_fig.update_traces(textposition="outside")
    st.plotly_chart(monthly_fig, use_container_width=False, width=800, height=400)

    st.markdown("**4️⃣ Composição do Custo Total** - Veja qual categoria (combustível, seguro, etc.) consome mais de cada modelo")
    bar_data = breakdown.melt(id_vars="Modelo", var_name="Categoria", value_name="Custo")
    bar_data["Rotulo"] = bar_data.apply(lambda row: cost_label(row["Categoria"], row["Custo"]), axis=1)
    bar_fig = px.bar(
        bar_data,
        x="Modelo",
        y="Custo",
        color="Categoria",
        text="Rotulo",
        title="💰 Composição do Custo Total no Período Analisado",
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
    st.plotly_chart(bar_fig, use_container_width=False, width=800, height=400)

    # Novo gráfico: Depreciação estimada dos carros
    st.subheader("📉 Depreciação Estimada dos Veículos")
    depreciation_data = []
    for car in cars:
        if car.pagamento != "Assinatura":
            # Assumir depreciação linear de 15% ao ano (valor aproximado)
            depreciation_rate = 0.15
            initial_value = car.valor_base if not car.carro_atual else current.valor_base
            for year in range(1, horizon + 1):
                depreciated_value = initial_value * (1 - depreciation_rate) ** year
                depreciation_data.append({
                    "Modelo": car.nome,
                    "Ano": year,
                    "Valor depreciado": depreciated_value
                })
    if depreciation_data:
        dep_df = pd.DataFrame(depreciation_data)
        dep_df["Rotulo"] = dep_df["Valor depreciado"].map(short_money)
        dep_fig = px.line(
            dep_df,
            x="Ano",
            y="Valor depreciado",
            color="Modelo",
            markers=True,
            text="Rotulo",
            title="📉 Valor Estimado do Veículo Após Depreciação (15% a.a.)",
        )
        dep_fig.update_traces(textposition="top center")
        st.plotly_chart(dep_fig, use_container_width=False, width=800, height=400)
    else:
        st.info("ℹ️ Não há carros com depreciação para exibir (apenas assinaturas não depreciam).")

    st.subheader("📈 Curva de Ponto de Virada (Break-Even)")
    st.caption("Explore como mudanças de preço e consumo afetariam a decisão. Valores em vermelho (negativos) = mais econômico que o carro atual.")
    target_options = {f"{car.nome} ({car.id})": car for car in cars if car.id != current.id}
    target_label = st.selectbox("🔍 Modelo para Testar contra o Carro Atual", list(target_options), help="Selecione um modelo alternativo para explorar qual preço/consumo o tornaria mais econômico.")
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
        title=f"📈 Preço e Consumo que Fariam {target.nome} Bater {current.nome}",
        xaxis_title="Preço do Veículo (R$)",
        yaxis_title=f"Consumo ({unit})",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Valores negativos indicam que o modelo testado fica mais barato que manter o carro atual.")


if __name__ == "__main__":
    main()
