

# Title: Compare Car
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
from datetime import datetime
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Imports para PDF são feitos dentro do botão, mas para verificação antecipada
# usamos try/except para não quebrar o app se não estiverem instalados

VehicleType = Literal["Combustao", "Hibrido", "Eletrico"]
PaymentType = Literal["A vista", "Financiado", "Assinatura"]


STATE_IPVA_DEFAULTS = {
    "AC": 2.0, "AL": 3.0, "AP": 3.0, "AM": 3.0, "BA": 2.5, "CE": 3.0, "DF": 3.5, "ES": 2.0,
    "GO": 3.75, "MA": 2.5, "MT": 3.0, "MS": 3.0, "MG": 4.0, "PA": 2.5, "PB": 2.5, "PR": 3.5,
    "PE": 3.0, "PI": 2.5, "RJ": 4.0, "RN": 3.0, "RS": 3.0, "RO": 3.0, "RR": 3.0, "SC": 2.0,
    "SP": 4.0, "SE": 3.0, "TO": 2.0,
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
    depreciation_factor: float = 15.0
    valor_carro_atual_trade_in: float = 0.0


def display_car_name(car: CarInputs) -> str:
    """Retorna o nome do carro com a forma de pagamento para exibição."""
    return f"{car.nome} - {car.pagamento}"


SIMULATIONS_DIR = Path("simulacoes")
SIMULATIONS_DIR.mkdir(exist_ok=True)


def is_visible_simulation_file(filepath: Path) -> bool:
    return filepath.suffix == ".json" and not filepath.stem.endswith("_DRAFT") and not filepath.stem.startswith("_AUTO_")


def save_simulation(name, ipva_default, horizon, investment_return, cars):
    data = {
        "timestamp": datetime.now().isoformat(),
        "ipva_default": ipva_default,
        "horizon": horizon,
        "investment_return": investment_return,
        "cars": [car.__dict__ for car in cars],
    }
    with open(SIMULATIONS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_simulation(name):
    with open(SIMULATIONS_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cars = []
    for car_data in data["cars"]:
        if "valor_carro_atual_trade_in" not in car_data:
            car_data["valor_carro_atual_trade_in"] = 0.0
        if "depreciation_factor" not in car_data:
            car_data["depreciation_factor"] = 15.0
        cars.append(CarInputs(**car_data))
    return data["ipva_default"], data["horizon"], data["investment_return"], cars


def get_simulations_metadata():
    simulations = []
    for filepath in sorted(SIMULATIONS_DIR.glob("*.json"), key=lambda x: os.path.getmtime(x), reverse=True):
        if not is_visible_simulation_file(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            cars_data = data.get("cars", [])
            model_names = [f"{car.get('nome', '?')} - {car.get('pagamento', '?')}" for car in cars_data]

            models_info = []
            for car in cars_data:
                current_value = car.get("valor_base", 0)
                monthly_cost = (
                    (current_value * car.get("ipva_percentual", 0) / 100 / 12) +
                    (car.get("seguro_anual", 0) / 12) +
                    (car.get("licenciamento_anual", 0) / 12) +
                    (car.get("manutencao_anual", 0) / 12) +
                    (car.get("pneus_anual", 0) / 12) +
                    (car.get("estacionamento_anual", 0) / 12) +
                    (car.get("outros_anuais", 0) / 12)
                )
                if car.get("pagamento") == "Assinatura":
                    monthly_cost = car.get("assinatura_mensal", 0)
                if car.get("pagamento") == "Financiado":
                    principal = max(car.get("valor_base", 0) - car.get("entrada_extra", 0), 0)
                    monthly_rate = car.get("taxa_juros_am", 0) / 100
                    months = car.get("prazo_meses", 1)
                    if principal > 0 and monthly_rate > 0 and months > 0:
                        parcela = principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
                    else:
                        parcela = principal / max(months, 1) if months > 0 else 0
                    monthly_cost += parcela
                models_info.append({
                    "nome": f"{car.get('nome', '?')} - {car.get('pagamento', '?')}",
                    "custo_mensal": monthly_cost,
                    "tipo": car.get("tipo", "?"),
                    "valor_base": car.get("valor_base", 0),
                })

            timestamp_str = data.get("timestamp", "")
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    formatted_time = dt.strftime("%d/%m/%Y %H:%M")
                except:
                    formatted_time = "Data desconhecida"
            else:
                formatted_time = "Data desconhecida"

            simulations.append({
                "filename": filepath.stem,
                "models": model_names,
                "models_info": models_info,
                "timestamp": formatted_time,
                "datetime_obj": datetime.fromisoformat(timestamp_str) if timestamp_str else None,
            })
        except Exception:
            continue
    return simulations


def list_simulations():
    return [f.stem for f in SIMULATIONS_DIR.glob("*.json") if is_visible_simulation_file(f)]


def format_simulation_title(models):
    return " vs ".join(models)


def save_draft(ipva_default, horizon, investment_return, cars):
    try:
        save_simulation("_DRAFT", ipva_default, horizon, investment_return, cars)
        if "last_auto_save_time" not in st.session_state:
            st.session_state.last_auto_save_time = datetime.now()
        current_time = datetime.now()
        time_elapsed = (current_time - st.session_state.last_auto_save_time).total_seconds()
        if time_elapsed >= 3600:
            existing_autos = list(SIMULATIONS_DIR.glob("_AUTO_*.json"))
            auto_number = len(existing_autos) + 1
            model_names = [car.nome for car in cars]
            model_str = "_".join(model_names).replace(" ", "_")[:50]
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            auto_name = f"_AUTO_{auto_number}_{model_str}_{timestamp}"
            save_simulation(auto_name, ipva_default, horizon, investment_return, cars)
            st.session_state.last_auto_save_time = current_time
    except Exception:
        pass


def load_draft():
    try:
        return load_simulation("_DRAFT")
    except Exception:
        return None


def clear_draft():
    try:
        (SIMULATIONS_DIR / "_DRAFT.json").unlink()
    except Exception:
        pass


def set_reload_notice(message: str) -> None:
    st.session_state.reload_notice = message


def render_reload_notice() -> None:
    message = st.session_state.get("reload_notice")
    if message:
        st.warning(message)


def copy_previous_car_state(index: int, source_car: CarInputs) -> None:
    trade_in = getattr(source_car, "valor_carro_atual_trade_in", 0.0)
    entry_pct = (source_car.entrada_extra / source_car.valor_base * 100) if source_car.valor_base > 0 else 0.0
    widget_values = {
        f"name_{index}": source_car.nome,
        f"type_{index}": source_car.tipo,
        f"value_{index}": source_car.valor_base,
        f"km_{index}": source_car.km_mes,
        f"pay_{index}": source_car.pagamento,
        f"trade_in_{index}": trade_in,
        f"ipva_{index}": source_car.ipva_percentual,
        f"km_l_{index}": source_car.consumo_km_l,
        f"fuel_{index}": source_car.preco_combustivel,
        f"ext_{index}": source_car.percentual_recarga_externa,
        f"km_kwh_{index}": source_car.consumo_km_kwh,
        f"kwh_{index}": source_car.preco_kwh,
        f"down_pct_{index}": entry_pct,
        f"rate_{index}": source_car.taxa_juros_am,
        f"term_{index}": source_car.prazo_meses,
        f"sub_m_{index}": source_car.assinatura_mensal,
        f"sub_fee_{index}": source_car.taxa_inicial_assinatura,
        f"depr_{index}": source_car.depreciation_factor,
        f"ins_{index}": source_car.seguro_anual,
        f"lic_{index}": source_car.licenciamento_anual,
        f"maint_{index}": source_car.manutencao_anual,
        f"tires_{index}": source_car.pneus_anual,
        f"park_{index}": source_car.estacionamento_anual,
        f"other_{index}": source_car.outros_anuais,
    }
    for key, value in widget_values.items():
        st.session_state[key] = value
    for y, rev in enumerate(source_car.revisoes_anuais, start=1):
        st.session_state[f"rev_{index}_{y}"] = rev


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


def monthly_money(value: float) -> str:
    return f"{money(value)}/mês"


def monthly_payment(principal, monthly_rate, months):
    if principal <= 0 or months <= 0:
        return 0.0
    if monthly_rate <= 0:
        return principal / months
    return principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)


def financing_principal(car: CarInputs) -> float:
    if car.carro_atual or car.pagamento != "Financiado":
        return 0.0
    return max(car.valor_base - car.entrada_extra, 0.0)


def financing_monthly_payment(car: CarInputs) -> float:
    return monthly_payment(financing_principal(car), car.taxa_juros_am / 100, car.prazo_meses)


def energy_cost(car: CarInputs) -> float:
    annual_km = car.km_mes * 12
    external_share = car.percentual_recarga_externa / 100 if car.tipo == "Hibrido" else 0
    if car.tipo == "Eletrico":
        return annual_km / max(car.consumo_km_kwh, 0.01) * car.preco_kwh
    if car.tipo == "Hibrido":
        gen_cost = annual_km * (1 - external_share) / max(car.consumo_km_l, 0.01) * car.preco_combustivel
        plug_cost = annual_km * external_share / max(car.consumo_km_kwh, 0.01) * car.preco_kwh
        return gen_cost + plug_cost
    return annual_km / max(car.consumo_km_l, 0.01) * car.preco_combustivel


def financing_schedule_by_year(car: CarInputs) -> list[dict[str, float]]:
    schedule = []
    principal = financing_principal(car)
    monthly_rate = car.taxa_juros_am / 100
    installment = monthly_payment(principal, monthly_rate, car.prazo_meses)
    balance = principal
    cumulative_amort = 0.0
    for year in range(1, car.horizonte_anos + 1):
        annual_payment = 0.0
        annual_interest = 0.0
        annual_amort = 0.0
        for month in range(1, 13):
            if (year - 1) * 12 + month > car.prazo_meses or balance <= 0:
                break
            interest = balance * monthly_rate
            amort = min(installment - interest, balance)
            payment = interest + amort
            balance = max(balance - amort, 0.0)
            annual_payment += payment
            annual_interest += interest
            annual_amort += amort
        cumulative_amort += annual_amort
        schedule.append({
            "Juros financiamento": annual_interest,
            "Amortizacao financiamento": annual_amort,
            "Valor quitado financiamento": car.entrada_extra + cumulative_amort,
            "Saldo financiamento": balance,
        })
    return schedule


def simulate_car(car: CarInputs, current_value: float, investment_return: float, fin_term_default: float = 24) -> pd.DataFrame:
    trade_in_value = getattr(car, 'valor_carro_atual_trade_in', 0.0)
    trade_in_value = trade_in_value if trade_in_value > 0 else current_value

    if car.carro_atual:
        additional_capital = 0.0
        opportunity_period_months = 0
    elif car.pagamento == "A vista":
        additional_capital = max(0.0, car.valor_base - trade_in_value)
        opportunity_period_months = fin_term_default
    elif car.pagamento == "Financiado":
        additional_capital = max(0.0, car.entrada_extra - trade_in_value)
        opportunity_period_months = car.prazo_meses
    else:
        additional_capital = 0.0
        opportunity_period_months = 0

    rate = investment_return / 100
    opportunity_years = opportunity_period_months / 12.0
    schedule = financing_schedule_by_year(car)

    rows = []
    remaining_years = opportunity_years
    accumulated = additional_capital

    for year in range(1, car.horizonte_anos + 1):
        if remaining_years > 0 and accumulated > 0:
            fraction = min(1.0, remaining_years)
            annual_opportunity = accumulated * ((1 + rate) ** fraction - 1)
            accumulated *= (1 + rate) ** fraction
            remaining_years -= fraction
        else:
            annual_opportunity = 0.0

        revision = car.revisoes_anuais[year - 1] if year <= len(car.revisoes_anuais) else 0
        subscription = (
            car.assinatura_mensal * 12 + (car.taxa_inicial_assinatura if year == 1 else 0)
            if car.pagamento == "Assinatura"
            else 0.0
        )
        ipva = 0.0 if car.pagamento == "Assinatura" else car.valor_base * car.ipva_percentual / 100
        lic = 0.0 if car.pagamento == "Assinatura" else car.licenciamento_anual
        seguro = 0.0 if car.pagamento == "Assinatura" else car.seguro_anual
        manut = 0.0 if car.pagamento == "Assinatura" else car.manutencao_anual
        pneus = 0.0 if car.pagamento == "Assinatura" else car.pneus_anual
        rev = 0.0 if car.pagamento == "Assinatura" else revision

        operating = (
            energy_cost(car) + ipva + lic + seguro + rev + manut + pneus
            + car.estacionamento_anual + car.outros_anuais + subscription
        )
        fin_year = schedule[year - 1]
        total = operating + annual_opportunity + fin_year["Juros financiamento"]

        rows.append({
            "Modelo": display_car_name(car),
            "ID": car.id,
            "Ano": year,
            "Combustível/energia": energy_cost(car),
            "IPVA": ipva,
            "Licenciamento": lic,
            "Seguro": seguro,
            "Revisões": rev,
            "Manutenção": manut,
            "Pneus": pneus,
            "Estacionamento": car.estacionamento_anual,
            "Outros": car.outros_anuais,
            "Assinatura": subscription,
            "Custo de oportunidade": annual_opportunity,
            "Juros financiamento": fin_year["Juros financiamento"],
            "Custo operacional total": operating,
            "Custo total anual": total,
        })
    return pd.DataFrame(rows)


def add_general_assumptions(ipva_default_val=4.0, horizon_val=3, investment_return_val=10.0,
                           fin_entry_pct_val=20.0, fin_rate_val=1.3, fin_term_val=24):
    st.subheader("📋 Premissas Gerais da Análise")
    col1, col2, col3 = st.columns(3)

    if "ipva_default" not in st.session_state:
        st.session_state.ipva_default = ipva_default_val
    if "previous_state" not in st.session_state:
        st.session_state.previous_state = "SP"

    state = col1.selectbox("🏠 Estado", sorted(STATE_IPVA_DEFAULTS),
                           index=sorted(STATE_IPVA_DEFAULTS).index(st.session_state.previous_state))
    if state != st.session_state.previous_state:
        st.session_state.ipva_default = STATE_IPVA_DEFAULTS[state]
        st.session_state.previous_state = state

    ipva_default = col1.number_input("📊 Alíquota Padrão de IPVA (% ao ano)", 0.0, 10.0,
                                     st.session_state.ipva_default, 0.1,
                                     help="Alíquota padrão de IPVA baseada no estado selecionado. "
                                          "Aplicada a todos os modelos alternativos por padrão.")
    st.session_state.ipva_default = ipva_default

    horizon = col2.slider("⏱️ Período de Análise (anos)", 1, 10, horizon_val,
                         help="Horizonte de tempo para a análise. Determina quantos anos de custos serão simulados.")
    investment_return = col3.number_input("💰 Retorno Esperado de Investimentos (% ao ano)", 0.0, 30.0,
                                          investment_return_val, 0.5,
                                          help="Rendimento que você obteria investindo o capital adicional. "
                                               "Usado para calcular o custo de oportunidade com juros compostos (proporcional ao período).")

    st.divider()
    st.markdown("#### 💳 Condições Padrão de Financiamento")
    st.caption("Valores padrão aplicados automaticamente quando 'Financiado' é selecionado. Personalizáveis por modelo.")
    fin_cols = st.columns(3)
    with fin_cols[0]:
        fin_entry_pct = fin_cols[0].number_input("📈 Entrada Mínima (%)", 0.0, 100.0, fin_entry_pct_val, 5.0,
                                                 help="Percentual mínimo de entrada para financiamento. Ex: 20% de entrada no valor do veículo.")
    with fin_cols[1]:
        fin_rate = fin_cols[1].number_input("📊 Taxa Padrão (% a.m.)", 0.0, 10.0, fin_rate_val, 0.05,
                                            help="Taxa de juros mensal padrão. Ex: 1.3% a.m. é aproximadamente 16.8% a.a.")
    with fin_cols[2]:
        fin_term = fin_cols[2].number_input("⏱️ Prazo Padrão (meses)", 1, 120, fin_term_val, 1,
                                            help="Prazo padrão do financiamento. Ex: 24 meses. "
                                                 "Para 'À vista', usado para calcular custo de oportunidade.")
    st.info(
        f"**Resumo**: Entrada {fin_entry_pct:.0f}% | Taxa {fin_rate:.2f}% a.m. | Prazo {fin_term:.0f} meses\n\n"
        f"Para modelos \"À vista\", o prazo de {fin_term:.0f} meses será usado para calcular o custo de oportunidade, "
        f"representando quanto você teria ganho se tivesse investido o capital adicional nesse período (juros compostos)."
    )
    return state, ipva_default, horizon, investment_return, fin_entry_pct, fin_rate, fin_term


def car_form(index: int, ipva_default: float, horizon: int, current_value: float | None = None,
             default_car: CarInputs | None = None, investment_return: float = 10.0,
             previous_car: CarInputs | None = None,
             fin_entry_pct_default: float = 20.0, fin_rate_default: float = 1.3,
             fin_term_default: float = 24) -> CarInputs:
    current = index == 0
    label = "Carro atual" if current else f"Modelo {index}"

    if st.session_state.pop(f"copy_from_{index}", False) and previous_car:
        copy_previous_car_state(index, previous_car)

    d_nome = default_car.nome if default_car else (st.session_state.get(f"name_{index}", previous_car.nome if previous_car else label))
    d_tipo = default_car.tipo if default_car else (st.session_state.get(f"type_{index}", previous_car.tipo if previous_car else "Combustao"))
    d_pagamento = default_car.pagamento if default_car else (st.session_state.get(f"pay_{index}", previous_car.pagamento if previous_car and not current else "A vista"))
    d_valor_base = default_car.valor_base if default_car else (st.session_state.get(f"value_{index}", previous_car.valor_base if previous_car else (122_000.0 if current else 225_000.0)))
    d_km = default_car.km_mes if default_car else (st.session_state.get(f"km_{index}", previous_car.km_mes if previous_car else 900.0))
    d_ipva = default_car.ipva_percentual if default_car else (st.session_state.get(f"ipva_{index}", previous_car.ipva_percentual if previous_car else ipva_default))
    d_seguro = default_car.seguro_anual if default_car else (st.session_state.get(f"ins_{index}", previous_car.seguro_anual if previous_car else (d_valor_base * 0.035)))
    d_lic = default_car.licenciamento_anual if default_car else (st.session_state.get(f"lic_{index}", previous_car.licenciamento_anual if previous_car else 200.0))
    d_manut = default_car.manutencao_anual if default_car else (st.session_state.get(f"maint_{index}", previous_car.manutencao_anual if previous_car else 1000.0))
    d_pneus = default_car.pneus_anual if default_car else (st.session_state.get(f"tires_{index}", previous_car.pneus_anual if previous_car else 0.0))
    d_estac = default_car.estacionamento_anual if default_car else (st.session_state.get(f"park_{index}", previous_car.estacionamento_anual if previous_car else 0.0))
    d_outros = default_car.outros_anuais if default_car else (st.session_state.get(f"other_{index}", previous_car.outros_anuais if previous_car else 0.0))
    d_revisoes = list(default_car.revisoes_anuais) if default_car else (list(previous_car.revisoes_anuais) if previous_car else [1200.0 + 250.0 * ((y - 1) % 3) for y in range(1, horizon + 1)])
    d_cons_l = default_car.consumo_km_l if default_car else (st.session_state.get(f"km_l_{index}", previous_car.consumo_km_l if previous_car else (6.5 if d_tipo == "Combustao" else 15.0)))
    d_cons_kwh = default_car.consumo_km_kwh if default_car else (st.session_state.get(f"km_kwh_{index}", previous_car.consumo_km_kwh if previous_car else 6.0))
    d_preco_comb = default_car.preco_combustivel if default_car else (st.session_state.get(f"fuel_{index}", previous_car.preco_combustivel if previous_car else 6.50))
    d_preco_kwh = default_car.preco_kwh if default_car else (st.session_state.get(f"kwh_{index}", previous_car.preco_kwh if previous_car else 1.30))
    d_recarga = default_car.percentual_recarga_externa if default_car else (st.session_state.get(f"ext_{index}", previous_car.percentual_recarga_externa if previous_car else 0.0))
    d_entrada = default_car.entrada_extra if default_car else (previous_car.entrada_extra if previous_car else 0.0)
    d_taxa = default_car.taxa_juros_am if default_car else (st.session_state.get(f"rate_{index}", previous_car.taxa_juros_am if previous_car else fin_rate_default))
    d_prazo = default_car.prazo_meses if default_car else (st.session_state.get(f"term_{index}", previous_car.prazo_meses if previous_car else int(fin_term_default)))
    d_assinatura_m = default_car.assinatura_mensal if default_car else (st.session_state.get(f"sub_m_{index}", previous_car.assinatura_mensal if previous_car else 3500.0))
    d_assinatura_tx = default_car.taxa_inicial_assinatura if default_car else (st.session_state.get(f"sub_fee_{index}", previous_car.taxa_inicial_assinatura if previous_car else 0.0))
    d_depr = default_car.depreciation_factor if default_car else (st.session_state.get(f"depr_{index}", previous_car.depreciation_factor if previous_car else 15.0))
    d_trade_in = getattr(default_car, 'valor_carro_atual_trade_in', 0.0) if default_car else (st.session_state.get(f"trade_in_{index}", getattr(previous_car, 'valor_carro_atual_trade_in', 0.0) if previous_car else (current_value if current_value else 0.0)))

    with st.expander(label, expanded=index <= 2):
        if not current and previous_car:
            col_copy, col_space = st.columns([0.3, 0.7])
            with col_copy:
                if st.button("📋 Copiar modelo anterior", key=f"copy_{index}", use_container_width=True):
                    copy_previous_car_state(index, previous_car)
                    set_reload_notice(
                        f"✅ O modelo anterior foi copiado para o Modelo {index}. Se algum campo não atualizar visualmente de imediato, pressione F5."
                    )
                    st.rerun()
            st.divider()
        st.markdown("##### 1. Dados principais")
        if current:
            main_cols = st.columns(4)
            with main_cols[0]:
                nome = st.text_input("🚗 Nome/Modelo", d_nome, key=f"name_{index}")
            with main_cols[1]:
                tipo = st.selectbox("⚡ Tipo", ["Combustao", "Hibrido", "Eletrico"],
                                    index=["Combustao", "Hibrido", "Eletrico"].index(d_tipo), key=f"type_{index}")
            with main_cols[2]:
                valor_base = st.number_input("💵 Valor de Mercado Atual", 0.0, 2_000_000.0, d_valor_base, 1000.0, key=f"value_{index}")
            with main_cols[3]:
                km_mes = st.number_input("📍 Quilometragem Mensal (km)", 0.0, 20_000.0, d_km, 100.0, key=f"km_{index}")
            pagamento = "A vista"
            trade_in = 0.0
            st.caption("O carro atual serve como referência de comparação. A forma de pagamento aqui é mantida só por compatibilidade com simulações antigas.")
        else:
            main_cols = st.columns(4)
            with main_cols[0]:
                nome = st.text_input("🚗 Nome/Modelo", d_nome, key=f"name_{index}")
            with main_cols[1]:
                tipo = st.selectbox("⚡ Tipo", ["Combustao", "Hibrido", "Eletrico"],
                                    index=["Combustao", "Hibrido", "Eletrico"].index(d_tipo), key=f"type_{index}")
            with main_cols[2]:
                pagamento = st.selectbox("💳 Forma de Pagamento", ["A vista", "Financiado", "Assinatura"],
                                         index=["A vista", "Financiado", "Assinatura"].index(d_pagamento),
                                         key=f"pay_{index}")
            with main_cols[3]:
                valor_base = st.number_input(
                    "💵 Preço do Modelo" if pagamento != "Assinatura" else "💵 Valor de Referência",
                    0.0, 2_000_000.0, d_valor_base, 1000.0, key=f"value_{index}")

            alt_cols = st.columns(2)
            with alt_cols[0]:
                km_mes = st.number_input("📍 Quilometragem Mensal (km)", 0.0, 20_000.0, d_km, 100.0, key=f"km_{index}")
            with alt_cols[1]:
                trade_in = st.number_input(
                    "💰 Valor de troca do seu carro atual (proposta)",
                    0.0, 2_000_000.0, d_trade_in, 1000.0, key=f"trade_in_{index}")
                if current_value:
                    st.caption(f"Padrão (seu carro): {money(current_value)}")

        st.divider()
        st.markdown("##### 2. Uso e energia")
        consumo_l = d_cons_l
        consumo_kwh = d_cons_kwh
        preco_combustivel = d_preco_comb
        preco_kwh = d_preco_kwh
        percentual_recarga_externa = d_recarga

        if tipo == "Combustao":
            uso_cols = st.columns(2)
            with uso_cols[0]:
                consumo_l = st.number_input("⛽ Consumo (km/l)", 0.1, 100.0, d_cons_l, 0.5, key=f"km_l_{index}")
            with uso_cols[1]:
                preco_combustivel = st.number_input("💰 Gasolina (R$/l)", 0.0, 20.0, d_preco_comb, 0.05, key=f"fuel_{index}")
        elif tipo == "Hibrido":
            uso_cols = st.columns(3)
            with uso_cols[0]:
                consumo_l = st.number_input("⛽ Consumo Motor (km/l equiv.)", 0.1, 100.0, d_cons_l, 0.5, key=f"km_l_{index}")
            with uso_cols[1]:
                preco_combustivel = st.number_input("💰 Gasolina (R$/l)", 0.0, 20.0, d_preco_comb, 0.05, key=f"fuel_{index}")
            with uso_cols[2]:
                percentual_recarga_externa = st.slider("🔌 Recarga externa (%)", 0.0, 100.0, d_recarga, 5.0, key=f"ext_{index}")
            if percentual_recarga_externa > 0:
                elet_cols = st.columns(2)
                with elet_cols[0]:
                    consumo_kwh = st.number_input("⚡ Eficiência (km/kWh)", 0.1, 20.0, d_cons_kwh, 0.1, key=f"km_kwh_{index}")
                with elet_cols[1]:
                    preco_kwh = st.number_input("💰 Tarifa energia (R$/kWh)", 0.0, 5.0, d_preco_kwh, 0.05, key=f"kwh_{index}")
        else:
            uso_cols = st.columns(2)
            with uso_cols[0]:
                consumo_kwh = st.number_input("⚡ Eficiência (km/kWh)", 0.1, 20.0, d_cons_kwh, 0.1, key=f"km_kwh_{index}")
            with uso_cols[1]:
                preco_kwh = st.number_input("💰 Tarifa energia (R$/kWh)", 0.0, 5.0, d_preco_kwh, 0.05, key=f"kwh_{index}")

        st.divider()
        st.markdown("##### 3. Compra e contratação")
        entrada_extra = 0.0
        taxa = d_taxa
        prazo = d_prazo
        assinatura_mensal = d_assinatura_m
        taxa_inicial_assinatura = d_assinatura_tx
        depr = d_depr

        if pagamento == "Financiado":
            fin_cols = st.columns(3)
            with fin_cols[0]:
                pct = (d_entrada / valor_base * 100) if valor_base > 0 else fin_entry_pct_default
                pct = st.slider("💵 Entrada (%)", 0.0, 100.0, pct, 5.0, key=f"down_pct_{index}")
                entrada_extra = valor_base * pct / 100
            with fin_cols[1]:
                taxa = st.number_input("📈 Juros mensais (%)", 0.0, 10.0, d_taxa, 0.05, key=f"rate_{index}")
            with fin_cols[2]:
                prazo = st.number_input("⏱️ Prazo (meses)", 1, 120, d_prazo, 1, key=f"term_{index}")
            parcela = financing_monthly_payment(
                CarInputs(id="", nome="", carro_atual=False, tipo="Combustao", valor_base=valor_base,
                          pagamento="Financiado", entrada_extra=entrada_extra, taxa_juros_am=taxa, prazo_meses=int(prazo),
                          assinatura_mensal=0, taxa_inicial_assinatura=0, km_mes=0, consumo_km_l=0, consumo_km_kwh=0,
                          preco_combustivel=0, preco_kwh=0, percentual_recarga_externa=0, ipva_percentual=0,
                          licenciamento_anual=0, seguro_anual=0, revisoes_anuais=(), manutencao_anual=0,
                          pneus_anual=0, estacionamento_anual=0, outros_anuais=0, horizonte_anos=horizon,
                          depreciation_factor=d_depr))
            st.success(f"💳 Parcela: {monthly_money(parcela)}")
            financed = valor_base - entrada_extra
            total_paid = parcela * prazo
            total_cost = entrada_extra + total_paid
            st.info(
                f"**Detalhes do financiamento:**\n\n"
                f"Preço do veículo: {money(valor_base)}\n\n"
                f"Entrada: {money(entrada_extra)} ({pct:.0f}%)\n\n"
                f"Valor financiado: {money(financed)}\n\n"
                f"Parcela: {monthly_money(parcela)} x {prazo} meses\n\n"
                f"Total pago em parcelas: {money(total_paid)}\n\n"
                f"**Custo total (entrada + parcelas): {money(total_cost)}**"
            )
            st.caption("📉 Depreciação (informativo)")
            depr = st.slider("Fator anual (%)", 0.0, 30.0, d_depr, 1.0, key=f"depr_{index}")

        elif pagamento == "Assinatura":
            st.caption("📅 Condições da Assinatura")
            sub_cols = st.columns(2)
            with sub_cols[0]:
                assinatura_mensal = st.number_input("📅 Mensalidade (R$/mês)", 0.0, 100_000.0, d_assinatura_m, 100.0, key=f"sub_m_{index}")
            with sub_cols[1]:
                taxa_inicial_assinatura = st.number_input("💳 Taxa inicial (R$)", 0.0, 100_000.0, d_assinatura_tx, 500.0, key=f"sub_fee_{index}")
            st.info("✅ **Inclusos na assinatura**: IPVA, seguro, licenciamento, manutenção corretiva e pneus.\n\n"
                   "❌ **Não inclusos**: Combustível/energia, estacionamento, multas e acessórios.")
        else:
            st.caption("📉 Depreciação")
            depr = st.slider("Fator anual (%)", 0.0, 30.0, d_depr, 1.0, key=f"depr_{index}")

        st.divider()
        st.markdown("##### 4. Custos anuais")
        if pagamento == "Assinatura":
            extra_cols = st.columns(2)
            manutencao = 0.0
            pneus = 0.0
            with extra_cols[0]:
                estacionamento = st.number_input("🅿️ Estacionamento/ano", 0.0, 100_000.0, d_estac, 100.0, key=f"park_{index}")
            with extra_cols[1]:
                outros = st.number_input("📦 Outros/ano", 0.0, 100_000.0, d_outros, 100.0, key=f"other_{index}")
            ipva = d_ipva
            seguro = 0.0
            licenciamento = 0.0
            revisoes = tuple(0.0 for _ in range(horizon))
        else:
            fixed_cols = st.columns(3)
            with fixed_cols[0]:
                ipva = st.number_input("📋 IPVA (% ao ano)", 0.0, 10.0, d_ipva, 0.1, key=f"ipva_{index}")
            with fixed_cols[1]:
                seguro = st.number_input("🛡️ Seguro anual (R$)", 0.0, 100_000.0, d_seguro, 250.0, key=f"ins_{index}")
            with fixed_cols[2]:
                licenciamento = st.number_input("📝 Licenciamento anual (R$)", 0.0, 10_000.0, d_lic, 50.0, key=f"lic_{index}")
            running_cols = st.columns(4)
            with running_cols[0]:
                manutencao = st.number_input("🔧 Manutenção/ano", 0.0, 100_000.0, d_manut, 100.0, key=f"maint_{index}")
            with running_cols[1]:
                pneus = st.number_input("⛩️ Pneus/ano", 0.0, 100_000.0, d_pneus, 100.0, key=f"tires_{index}")
            with running_cols[2]:
                estacionamento = st.number_input("🅿️ Estacionamento/ano", 0.0, 100_000.0, d_estac, 100.0, key=f"park_{index}")
            with running_cols[3]:
                outros = st.number_input("📦 Outros/ano", 0.0, 100_000.0, d_outros, 100.0, key=f"other_{index}")
            st.caption("💰 Revisões por ano")
            rev_cols = st.columns(min(horizon, 5))
            revisoes = []
            for y in range(1, horizon + 1):
                col = rev_cols[(y - 1) % len(rev_cols)]
                revisoes.append(col.number_input(
                    f"Ano {y}",
                    0.0, 100_000.0,
                    d_revisoes[y - 1] if y <= len(d_revisoes) else 1200.0 + 250.0 * ((y - 1) % 3),
                    100.0,
                    key=f"rev_{index}_{y}"
                ))
            media_revisoes = sum(revisoes) / len(revisoes) if revisoes else 0.0
            st.info(f"📊 **Média anual de revisões no período: {money(media_revisoes)}**")

        if not current:
            trade_in_valor = trade_in if trade_in > 0 else (current_value if current_value else 0.0)
            if pagamento == "A vista":
                cap_adicional = max(0.0, valor_base - trade_in_valor)
            elif pagamento == "Financiado":
                cap_adicional = max(0.0, entrada_extra - trade_in_valor)
            else:
                cap_adicional = 0.0
            if cap_adicional > 0:
                st.warning(f"💸 **Capital Adicional Necessário: {money(cap_adicional)}**\n\n"
                           f"Esse valor precisará ser desembolsado (além da troca do seu carro) e gerará um custo de oportunidade anual de {investment_return:.1f}%.")
            else:
                st.success("✅ **Nenhum capital adicional necessário** (a proposta de troca cobre o desembolso).")

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
        depreciation_factor=depr,
        valor_carro_atual_trade_in=trade_in,
    )


def make_break_even(current: CarInputs, target: CarInputs, investment_return: float, fin_term_default: float = 24) -> pd.DataFrame:
    target_prices = np.linspace(max(target.valor_base * 0.6, 10_000), target.valor_base * 1.4, 30)
    if target.tipo == "Eletrico":
        consumptions = np.linspace(3.0, 10.0, 30)
    else:
        consumptions = np.linspace(5, 25, 30)
    current_annual = simulate_car(current, current.valor_base, investment_return, fin_term_default)["Custo total anual"].mean()
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
            target_annual = simulate_car(candidate, current.valor_base, investment_return, fin_term_default)["Custo total anual"].mean()
            diff = target_annual - current_annual
            rows.append({
                "Valor do carro": price,
                "Consumo": consumption,
                "Diferenca anual vs atual": diff,
            })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Comparador de Custo de Manter Veículos", layout="wide")
    st.title("🚗 Comparador de Custo de Manter Veículos")

    # ==================== SIDEBAR ====================
    st.sidebar.title("⚙️ Menu")
    if st.sidebar.button("🔄 Resetar Tudo\n(depois aperte F5)", use_container_width=True):
        clear_draft()
        st.cache_data.clear()
        st.cache_resource.clear()
        keys_to_delete = ["ipva_default", "horizon", "investment_return", "cars",
                        "draft_restored", "previous_state", "save_name_input",
                        "last_auto_save_time", "skip_draft_restore"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        if "reload_notice" in st.session_state:
            del st.session_state["reload_notice"]
        widget_patterns = ["name_", "pay_", "type_", "value_", "km_",
                          "ipva_", "km_l_", "fuel_", "ext_", "km_kwh_",
                          "kwh_", "down_pct_", "rate_", "term_", "ins_",
                          "maint_", "tires_", "park_", "other_", "rev_",
                          "trade_in_", "sub_", "depr_", "copy_from_",
                          "lic_", "copy_", "load_recent_", "load_sidebar_",
                          "manage", "be_select", "btn_"]
        widget_pattern_keys = [k for k in list(st.session_state.keys())
                             if any(pattern in k for pattern in widget_patterns)]
        for key in widget_pattern_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.skip_draft_restore = True
        st.warning("✅ Dados deletados! Agora aperte **F5** para recarregar a página e ver as alterações.")

    st.sidebar.markdown("### 📂 Últimas Simulações")
    simulations = get_simulations_metadata()
    if simulations:
        recent = simulations[:10]
        for sim in recent:
            models_display = format_simulation_title(sim['models'])
            with st.sidebar.expander(f"📅 {sim['timestamp']}\n{models_display}", expanded=False):
                st.write(f"**Modelos:** {', '.join(sim['models'])}")
                if st.button(f"📂 Carregar", key=f"load_sidebar_{sim['filename']}", use_container_width=True):
                    try:
                        ipva_default, horizon, investment_return, cars_temp = load_simulation(sim['filename'])
                        st.session_state.ipva_default = ipva_default
                        st.session_state.horizon = horizon
                        st.session_state.investment_return = investment_return
                        st.session_state.cars = cars_temp
                        st.session_state.draft_restored = True
                        set_reload_notice(
                            f"✅ A simulação '{sim['filename']}' foi carregada. Agora pressione F5 para recarregar a página e aplicar os campos."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar: {e}")
    else:
        st.sidebar.info("Nenhuma simulação salva ainda.")

    st.sidebar.divider()
    skip_restore = st.session_state.get("skip_draft_restore", False)
    if skip_restore:
        del st.session_state.skip_draft_restore
        st.info("🧹 Todos os dados foram limpos. Comece do zero!")
    else:
        draft = load_draft()
        if draft and not st.session_state.get("draft_restored"):
            ipva_d, h, ir, cars_d = draft
            st.session_state.ipva_default = ipva_d
            st.session_state.horizon = h
            st.session_state.investment_return = ir
            st.session_state.cars = cars_d
            st.session_state.draft_restored = True
            st.info("✅ Rascunho restaurado.")

    render_reload_notice()

    ipva_default_val = st.session_state.get("ipva_default", 4.0)
    horizon_val = st.session_state.get("horizon", 3)
    investment_return_val = st.session_state.get("investment_return", 10.0)
    cars_loaded = st.session_state.get("cars", [])

    state, ipva_default, horizon, investment_return, fin_entry_pct, fin_rate, fin_term = add_general_assumptions(ipva_default_val, horizon_val, investment_return_val)

    st.subheader("🧠 Premissas da Análise")
    st.write(
        "**Custo de oportunidade**: incide apenas sobre o **capital adicional** durante o período em que ele está sendo utilizado:\n\n"
        "- **À vista**: o capital inteiro é desembolsado no ano 1, gerando custo de oportunidade pelo **prazo padrão de financiamento** "
        f"(atualmente {int(fin_term)} meses). Isso representa quanto você teria ganho se tivesse investido esse capital.\n\n"
        "- **Financiado**: apenas a **entrada** gera custo de oportunidade, e apenas durante o **prazo do financiamento do modelo**. "
        "Após o financiamento ser quitado, não há mais custo. O custo é calculado com **juros compostos** durante o período.\n\n"
        "- **Assinatura**: não há capital adicional, portanto **não há custo de oportunidade**.\n\n"
        "Se o carro atual cobre a entrada (financiamento) ou o preço total (à vista), o custo de oportunidade é zero."
    )

    current = car_form(0, ipva_default, horizon, default_car=cars_loaded[0] if cars_loaded else None,
                       investment_return=investment_return,
                       fin_entry_pct_default=fin_entry_pct, fin_rate_default=fin_rate, fin_term_default=fin_term)
    n_models = st.number_input("💶 Quantos modelos alternativos comparar?", 1, 20,
                               value=len(cars_loaded) - 1 if cars_loaded else 2, step=1)
    cars = [current]
    for i in range(1, int(n_models) + 1):
        previous_car = cars[i - 1] if i > 0 else None
        car = car_form(i, ipva_default, horizon, current.valor_base,
                      default_car=cars_loaded[i] if cars_loaded and i < len(cars_loaded) else None,
                      investment_return=investment_return,
                      fin_entry_pct_default=fin_entry_pct, fin_rate_default=fin_rate, fin_term_default=fin_term,
                      previous_car=previous_car)
        cars.append(car)

    save_draft(ipva_default, horizon, investment_return, cars)

    with st.expander("💾 Salvar / Gerenciar Simulações", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            model_names = [car.nome for car in cars if car.nome.strip()]
            default_name = "_".join(model_names).replace(" ", "_")
            if "save_name_input" not in st.session_state:
                st.session_state.save_name_input = default_name
            if st.button("🔄 Gerar nome a partir dos modelos"):
                st.session_state.save_name_input = default_name
                st.rerun()
            name = st.text_input("Nome da simulação", key="save_name_input", value=st.session_state.save_name_input)
            if st.button("💾 Salvar", key="btn_save") and name:
                try:
                    save_simulation(name, ipva_default, horizon, investment_return, cars)
                    st.success(f"Simulação '{name}' salva!")
                except Exception as e:
                    st.error(f"Erro: {e}")
        with col_b:
            sim_names = list_simulations()
            if sim_names:
                manage = st.selectbox("Gerenciar", [""] + sim_names, key="manage")
                if manage:
                    if st.button("🗑️ Excluir", key="btn_del"):
                        (SIMULATIONS_DIR / f"{manage}.json").unlink()
                        st.success(f"'{manage}' excluída!")
                        st.rerun()

    # Simulação
    sim_df = pd.concat([simulate_car(car, current.valor_base, investment_return, fin_term_default=fin_term) for car in cars], ignore_index=True)
    annual_avg = sim_df.groupby(["Modelo", "ID"])["Custo total anual"].mean().reset_index()
    annual_avg["Custo mensal médio"] = annual_avg["Custo total anual"] / 12
    annual_avg = annual_avg.sort_values("Custo total anual")
    winner = annual_avg.iloc[0]
    model_order = annual_avg["Modelo"].tolist()

    st.subheader("🎯 Melhor Opção")
    st.metric(label=f"✅ {winner['Modelo']}", value=money(winner["Custo total anual"]), help="Custo anual médio")
    st.caption("Custo anual médio considerado ao longo do período da análise (já com juros compostos no custo de oportunidade).")

    # Tabela resumo com destaque no vencedor
    # Função para colorir a linha do vencedor
    def highlight_winner(s):
        return ['background-color: #c6efce' if s.Modelo == winner['Modelo'] else '' for _ in range(len(s))]
    styled_annual = annual_avg[["Modelo", "Custo total anual", "Custo mensal médio"]].style \
        .format({"Custo total anual": money, "Custo mensal médio": money}) \
        .apply(highlight_winner, axis=1)
    st.dataframe(styled_annual, use_container_width=True)

    st.subheader("💼 Custo de Oportunidade Detalhado")
    st.caption(
        "O custo de oportunidade é calculado apenas durante o período em que o capital adicional está sendo utilizado, "
        "agora corretamente para qualquer número de meses (inclusive prazos menores que 1 ano). "
        "Com juros compostos, o custo acumula ao longo dos anos."
    )
    opp_data = []
    for car in cars:
        if car.carro_atual:
            cap_adicional = 0.0
            trade_in_value = 0.0
        else:
            trade_in_value = getattr(car, 'valor_carro_atual_trade_in', 0.0)
            trade_in_value = trade_in_value if trade_in_value > 0 else current.valor_base
            if car.pagamento == "A vista":
                cap_adicional = max(0.0, car.valor_base - trade_in_value)
            elif car.pagamento == "Financiado":
                cap_adicional = max(0.0, car.entrada_extra - trade_in_value)
            else:
                cap_adicional = 0.0
        car_rows = sim_df[sim_df["ID"] == car.id]
        opp_total_periodo = float(car_rows["Custo de oportunidade"].sum()) if not car_rows.empty else 0.0
        opp_medio_anual = opp_total_periodo / max(horizon, 1)
        opp_medio_mensal = opp_medio_anual / 12
        opp_data.append({
            "Modelo": display_car_name(car),
            "Valor de Troca": trade_in_value,
            "Capital Adicional": cap_adicional,
            "Custo Oport. Total (Empatado)": opp_total_periodo,
            "Custo Oport. Médio Anual": opp_medio_anual,
            "Custo Oport. Médio Mensal": opp_medio_mensal,
        })
    opp_df = pd.DataFrame(opp_data)
    st.dataframe(
        opp_df.style.format({
            "Valor de Troca": money,
            "Capital Adicional": money,
            "Custo Oport. Total (Empatado)": money,
            "Custo Oport. Médio Anual": money,
            "Custo Oport. Médio Mensal": money,
        }),
        use_container_width=True,
    )
    st.caption("O custo total anual médio já incorpora o custo de oportunidade médio anual. "
               "A tabela acima mostra o valor acumulado no horizonte e sua média anual/mensal.")

    st.subheader("📊 Detalhamento do Custo Anual Médio por Modelo")
    categories = ["Combustível/energia", "IPVA", "Licenciamento", "Seguro",
                  "Revisões (média anual)", "Manutenção", "Pneus", "Estacionamento", "Outros",
                  "Assinatura", "Custo de oportunidade", "Juros financiamento"]
    breakd = sim_df.groupby("Modelo")[["Combustível/energia", "IPVA", "Licenciamento", "Seguro",
                                       "Revisões", "Manutenção", "Pneus", "Estacionamento", "Outros",
                                       "Assinatura", "Custo de oportunidade", "Juros financiamento"]].mean().reset_index()
    breakd = breakd.rename(columns={"Revisões": "Revisões (média anual)"})
    breakd = breakd.set_index("Modelo").loc[model_order].reset_index()
    detailed = annual_avg[["Modelo", "Custo total anual", "Custo mensal médio"]].merge(breakd, on="Modelo")
    col_order = ["Modelo", "Custo total anual", "Custo mensal médio"] + categories
    detailed = detailed[col_order]

    # Aplicar cores: totalizadores (Custo total anual, Custo mensal médio) -> verde claro; componentes -> azul claro
    total_cols = ["Custo total anual", "Custo mensal médio"]
    component_cols = [c for c in col_order if c not in total_cols and c != "Modelo"]
    def color_columns(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in total_cols:
            if col in styles.columns:
                styles[col] = 'background-color: #c6efce'
        for col in component_cols:
            if col in styles.columns:
                styles[col] = 'background-color: #bdd7ee'
        return styles
    styled_detailed = detailed.style \
        .format({col: money for col in col_order if col != "Modelo"}) \
        .apply(color_columns, axis=None)
    st.dataframe(styled_detailed, use_container_width=True)
    st.caption(
        "🟦 Componentes do custo 🟩 Totalizadores (Custo total anual e Custo mensal médio)\n\n"
        "O **Custo mensal médio** é obtido dividindo o **Custo total anual médio** por 12. "
        "Ele representa a média mensal de todos os custos operacionais, financeiros e de oportunidade ao longo do horizonte da simulação."
    )

    # ---------- Gráficos ----------
    st.subheader("📈 Gráficos")
    custom_palette = px.colors.qualitative.Plotly
    fig1 = px.bar(annual_avg, x="Modelo", y="Custo total anual",
                text=annual_avg["Custo total anual"].map(money),
                color="Modelo", title="Custo Anual Médio por Modelo",
                color_discrete_sequence=custom_palette,
                category_orders={"Modelo": model_order})
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=450, width=800)
    st.plotly_chart(fig1, use_container_width=False, width=800)

    bar_data = breakd.melt("Modelo", var_name="Categoria", value_name="Custo")
    bar_data["Rótulo"] = bar_data.apply(lambda r: f"{r['Categoria']}: {short_money(r['Custo'])}", axis=1)
    fig2 = px.bar(bar_data, x="Modelo", y="Custo", color="Categoria", text="Rótulo",
                title="Composição Média Anual dos Custos",
                color_discrete_sequence=custom_palette,
                category_orders={"Modelo": model_order})
    fig2.update_traces(textposition="inside", insidetextanchor="middle",
                    textfont=dict(size=10, color="black", family="sans-serif"))
    fig2.update_layout(
        height=540,
        width=900,
        margin=dict(t=150),
        title=dict(y=0.94, x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig2, use_container_width=False, width=900)

    dep_data = []
    for car in cars:
        if car.pagamento != "Assinatura":
            rate = car.depreciation_factor / 100
            val = car.valor_base if not car.carro_atual else current.valor_base
            for y in range(1, horizon + 1):
                dep_data.append({"Modelo": display_car_name(car), "Ano": y, "Valor": val * (1 - rate) ** y})
    if dep_data:
        dep_df = pd.DataFrame(dep_data)
        dep_df["Rotulo"] = dep_df["Valor"].map(short_money)
        fig3 = px.line(dep_df, x="Ano", y="Valor", color="Modelo", markers=True, text="Rotulo",
                    title="Evolução do valor do veículo",
                    color_discrete_sequence=custom_palette)
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=450, width=800)
        st.plotly_chart(fig3, use_container_width=False, width=800)

    st.subheader("📈 Ponto de Virada (Break-Even)")
    st.markdown("Explore como preço e consumo afetariam a decisão. Valores negativos (azul) = modelo alternativo mais barato que o atual.")
    target_options = {display_car_name(car): car for car in cars if car.id != current.id}
    if target_options:
        target_label = st.selectbox("🔍 Modelo para Testar contra o Carro Atual", list(target_options.keys()), key="be_select")
        target = target_options[target_label]
        with st.spinner("Gerando malha de break‑even..."):
            be_df = make_break_even(current, target, investment_return, fin_term_default=fin_term)
        unit = "km/kWh" if target.tipo == "Eletrico" else "km/l"
        fig_be = go.Figure(data=go.Contour(
            x=be_df["Valor do carro"],
            y=be_df["Consumo"],
            z=be_df["Diferenca anual vs atual"],
            colorscale="RdYlGn_r",
            contours=dict(showlabels=True),
            colorbar=dict(title="Diferença anual (R$)"),
        ))
        fig_be.add_trace(go.Scatter(
            x=[target.valor_base],
            y=[target.consumo_km_kwh if target.tipo == "Eletrico" else target.consumo_km_l],
            mode="markers",
            marker=dict(size=14, color="black", symbol="x"),
            name=f"{display_car_name(target)} atual",
        ))
        fig_be.update_layout(
            title=f"Break‑even: {display_car_name(target)} vs {display_car_name(current)}",
            xaxis_title="Preço do veículo (R$)",
            yaxis_title=f"Consumo ({unit})",
            height=600,
            width=850,
        )
        st.plotly_chart(fig_be, use_container_width=False, width=850)
        st.caption("Cores frias (azul/verde) indicam que o modelo seria mais barato que o carro atual. "
                   "O marcador preto mostra a premissa atual do modelo.")
    else:
        st.info("Adicione ao menos um modelo alternativo para ativar o break‑even.")

    # ================== BOTÃO PDF ==================
    st.subheader("📄 Relatório em PDF")
    try:
        import reportlab
        import kaleido
        pdf_available = True
    except ImportError as e:
        pdf_available = False
        missing = str(e).split("'")[1] if "'" in str(e) else "reportlab ou kaleido"
        st.warning(f"⚠️ Para gerar o PDF, instale as bibliotecas necessárias:\n\n```\npip install reportlab kaleido\n```\n\nFaltando: `{missing}`")

    if pdf_available:
        if st.button("📄 Gerar Relatório PDF", key="btn_pdf"):
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm
            import plotly.io as pio
            import io

            with st.spinner("Gerando relatório PDF... Isso pode levar alguns segundos."):
                def unit_price(value: float) -> str:
                    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                winner_name = annual_avg.iloc[0]["Modelo"]
                winner_car = next(car for car in cars if display_car_name(car) == winner_name)
                other_cars = [car for car in cars if display_car_name(car) != winner_name]

                # Gráficos para PDF
                fig1_pdf = px.bar(annual_avg, x="Modelo", y="Custo total anual",
                                 text=annual_avg["Custo total anual"].map(money),
                                 color="Modelo", title="Custo Anual Médio por Modelo",
                                 color_discrete_sequence=custom_palette,
                                 category_orders={"Modelo": model_order})
                fig1_pdf.update_traces(textposition="outside")
                fig1_pdf.update_layout(height=450, width=800)

                bar_data_pdf = breakd.melt("Modelo", var_name="Categoria", value_name="Custo")
                bar_data_pdf["Rótulo"] = bar_data_pdf.apply(lambda r: f"{r['Categoria']}: {short_money(r['Custo'])}", axis=1)
                fig2_pdf = px.bar(bar_data_pdf, x="Modelo", y="Custo", color="Categoria", text="Rótulo",
                                 title="Composição Média Anual dos Custos",
                                 color_discrete_sequence=custom_palette,
                                 category_orders={"Modelo": model_order})
                fig2_pdf.update_traces(textposition="inside", insidetextanchor="middle",
                                      textfont=dict(size=10, color="black"))
                fig2_pdf.update_layout(
                    height=540,
                    width=900,
                    margin=dict(t=150),
                    title=dict(y=0.94, x=0.5, xanchor="center"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
                )

                dep_data_pdf = []
                for car in cars:
                    if car.pagamento != "Assinatura":
                        rate = car.depreciation_factor / 100
                        val = car.valor_base if not car.carro_atual else current.valor_base
                        for y in range(1, horizon + 1):
                            dep_data_pdf.append({"Modelo": display_car_name(car), "Ano": y, "Valor": val * (1 - rate) ** y})
                if dep_data_pdf:
                    dep_df_pdf = pd.DataFrame(dep_data_pdf)
                    fig3_pdf = px.line(dep_df_pdf, x="Ano", y="Valor", color="Modelo", markers=True,
                                      title="Evolução do valor do veículo",
                                      color_discrete_sequence=custom_palette)
                    fig3_pdf.update_layout(height=450, width=800)
                else:
                    fig3_pdf = None

                def fig_to_image(fig, width=800):
                    if fig is None:
                        return None
                    try:
                        img_bytes = pio.to_image(fig, format="png", width=width, scale=2)
                        return img_bytes
                    except Exception as e:
                        st.error(f"Erro ao converter gráfico: {e}")
                        return None

                img1 = fig_to_image(fig1_pdf)
                img2 = fig_to_image(fig2_pdf)
                img3 = fig_to_image(fig3_pdf) if fig3_pdf else None

                be_images = []
                for other in other_cars:
                    be_df = make_break_even(winner_car, other, investment_return, fin_term_default=fin_term)
                    unit = "km/kWh" if other.tipo == "Eletrico" else "km/l"
                    fig_be = go.Figure(data=go.Contour(
                        x=be_df["Valor do carro"],
                        y=be_df["Consumo"],
                        z=be_df["Diferenca anual vs atual"],
                        colorscale="RdYlGn_r",
                        contours=dict(showlabels=True),
                        colorbar=dict(title="Diferença anual (R$)"),
                    ))
                    fig_be.add_trace(go.Scatter(
                        x=[other.valor_base],
                        y=[other.consumo_km_kwh if other.tipo == "Eletrico" else other.consumo_km_l],
                        mode="markers",
                        marker=dict(size=14, color="black", symbol="x"),
                        name=f"{display_car_name(other)} atual",
                    ))
                    fig_be.update_layout(
                        title=f"Break‑even: {display_car_name(other)} vs {display_car_name(winner_car)}",
                        xaxis_title="Preço do veículo (R$)",
                        yaxis_title=f"Consumo ({unit})",
                        height=500,
                        width=800,
                    )
                    be_img = fig_to_image(fig_be, width=800)
                    if be_img:
                        be_images.append((display_car_name(other), be_img))

                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
                styles = getSampleStyleSheet()
                title_style = styles["Title"]
                heading_style = styles["Heading2"]
                normal_style = styles["Normal"]
                story = []

                def add_kv_section(title, rows, col_widths=(6 * cm, 10 * cm)):
                    story.append(Paragraph(title, styles["Heading3"]))
                    if not rows:
                        story.append(Paragraph("Sem dados para exibir.", normal_style))
                        story.append(Spacer(1, 0.2 * cm))
                        return
                    table = Table([[k, v] for k, v in rows], colWidths=list(col_widths))
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.3 * cm))

                story.append(Paragraph("Relatório de Comparação de Custos de Veículos", title_style))
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("Premissas Gerais", heading_style))
                story.append(Paragraph(f"Estado para IPVA padrão: {state} (alíquota {ipva_default:.1f}%)", normal_style))
                story.append(Paragraph(f"Período de análise: {horizon} anos", normal_style))
                story.append(Paragraph(f"Retorno esperado de investimentos: {investment_return:.1f}% a.a.", normal_style))
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Modelos Analisados", heading_style))
                model_table_data = [["Modelo", "Tipo", "Pagamento", "Valor (R$)", "Km/mês"]]
                for car in cars:
                    model_table_data.append([
                        display_car_name(car),
                        car.tipo,
                        car.pagamento,
                        money(car.valor_base),
                        f"{car.km_mes:.0f}"
                    ])
                model_table = Table(model_table_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 3.5*cm, 2.5*cm])
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 6),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ]))
                story.append(model_table)
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Dados Completos de Entrada", heading_style))
                for car in cars:
                    display_name = display_car_name(car)
                    story.append(Paragraph(f"<b>{display_name}</b>", styles["Heading3"]))
                    main_rows = [
                        ["Tipo", car.tipo],
                        ["Forma de pagamento", car.pagamento],
                        ["Valor base", money(car.valor_base)],
                        ["Km/mês", f"{car.km_mes:.0f}"],
                    ]
                    if car.carro_atual:
                        main_rows.append(["Valor do carro atual", money(car.valor_base)])
                    else:
                        trade_offer = getattr(car, "valor_carro_atual_trade_in", 0.0) or current.valor_base
                        main_rows.append(["Valor oferecido pelo seu carro", money(trade_offer)])
                        if car.pagamento == "A vista":
                            cap_adicional = max(0.0, car.valor_base - trade_offer)
                        elif car.pagamento == "Financiado":
                            cap_adicional = max(0.0, car.entrada_extra - trade_offer)
                        else:
                            cap_adicional = 0.0
                        main_rows.append(["Capital adicional necessário", money(cap_adicional) if cap_adicional > 0 else "Nenhum"])
                    add_kv_section("Dados principais", main_rows)

                    usage_rows = []
                    if car.tipo == "Combustao":
                        usage_rows = [
                            ["Consumo", f"{car.consumo_km_l:.1f} km/l"],
                            ["Preço combustível", f"{unit_price(car.preco_combustivel)}/l"],
                        ]
                    elif car.tipo == "Eletrico":
                        usage_rows = [
                            ["Consumo", f"{car.consumo_km_kwh:.1f} km/kWh"],
                            ["Preço energia", f"{unit_price(car.preco_kwh)}/kWh"],
                        ]
                    else:
                        usage_rows = [
                            ["Consumo gasolina", f"{car.consumo_km_l:.1f} km/l"],
                            ["Preço gasolina", f"{unit_price(car.preco_combustivel)}/l"],
                            ["Recarga externa", f"{car.percentual_recarga_externa:.0f}%"],
                            ["Consumo elétrico", f"{car.consumo_km_kwh:.1f} km/kWh"],
                            ["Preço energia", f"{unit_price(car.preco_kwh)}/kWh"],
                        ]
                    add_kv_section("Uso e energia", usage_rows)

                    contract_rows = []
                    if car.pagamento == "Financiado":
                        parcela = monthly_payment(max(car.valor_base - car.entrada_extra, 0.0), car.taxa_juros_am / 100, car.prazo_meses)
                        contract_rows = [
                            ["Entrada extra", money(car.entrada_extra)],
                            ["Taxa juros mensal", f"{car.taxa_juros_am:.2f}%"],
                            ["Prazo (meses)", f"{car.prazo_meses}"],
                            ["Parcela estimada", monthly_money(parcela)],
                            ["Depreciação anual", f"{car.depreciation_factor:.1f}%"],
                        ]
                    elif car.pagamento == "Assinatura":
                        contract_rows = [
                            ["Assinatura mensal", money(car.assinatura_mensal)],
                            ["Taxa inicial", money(car.taxa_inicial_assinatura)],
                            ["IPVA / seguro / licenciamento", "Inclusos"],
                            ["Manutenção e pneus", "Inclusos"],
                        ]
                    else:
                        contract_rows = [["Depreciação anual", f"{car.depreciation_factor:.1f}%"]]
                    add_kv_section("Compra e contratação", contract_rows)

                    annual_rows = []
                    if car.pagamento == "Assinatura":
                        annual_rows = [
                            ["IPVA", "Incluso"],
                            ["Seguro anual", "Incluso"],
                            ["Licenciamento anual", "Incluso"],
                            ["Manutenção anual", "Incluso"],
                            ["Pneus anual", "Incluso"],
                            ["Estacionamento anual", money(car.estacionamento_anual)],
                            ["Outros anuais", money(car.outros_anuais)],
                            ["Revisões médias anuais", "Inclusas"],
                        ]
                    else:
                        media_rev = sum(car.revisoes_anuais) / len(car.revisoes_anuais) if car.revisoes_anuais else 0
                        annual_rows = [
                            ["IPVA (%)", f"{car.ipva_percentual:.1f}%"],
                            ["Seguro anual", money(car.seguro_anual)],
                            ["Licenciamento anual", money(car.licenciamento_anual)],
                            ["Manutenção anual", money(car.manutencao_anual)],
                            ["Pneus anual", money(car.pneus_anual)],
                            ["Estacionamento anual", money(car.estacionamento_anual)],
                            ["Outros anuais", money(car.outros_anuais)],
                            ["Revisões médias anuais", money(media_rev)],
                        ]
                    add_kv_section("Custos anuais", annual_rows)
                story.append(PageBreak())

                story.append(Paragraph("Resultados da Simulação", heading_style))
                story.append(Paragraph("Resumo do custo de oportunidade", heading_style))
                opp_summary = opp_df[[
                    "Modelo",
                    "Capital Adicional",
                    "Custo Oport. Total (Empatado)",
                    "Custo Oport. Médio Anual",
                    "Custo Oport. Médio Mensal",
                ]].copy()
                for col in opp_summary.columns[1:]:
                    opp_summary[col] = opp_summary[col].apply(money)
                opp_summary_data = [list(opp_summary.columns)] + opp_summary.values.tolist()
                opp_summary_table = Table(opp_summary_data, repeatRows=1, colWidths=[4 * cm, 3.5 * cm, 4 * cm, 3.5 * cm, 3.5 * cm])
                opp_summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('FONTSIZE', (0,0), (-1,-1), 7),
                ]))
                story.append(opp_summary_table)
                story.append(Spacer(1, 0.4*cm))

                story.append(Paragraph("Custo Anual Médio e Mensal", heading_style))
                comp_data = [["Modelo", "Custo Total Anual Médio", "Custo Mensal Médio"]]
                for _, row in annual_avg.iterrows():
                    comp_data.append([row["Modelo"], money(row["Custo total anual"]), money(row["Custo mensal médio"])])
                # Aumentar coluna Modelo para 6 cm, as outras 6 cm cada (total 18 cm)
                comp_table = Table(comp_data, colWidths=[6*cm, 6*cm, 6*cm])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    # Destaque para o vencedor (primeira linha de dados)
                    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#c6efce')),
                ]))
                story.append(comp_table)
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Detalhamento dos Custos Médios Anuais", heading_style))
                detailed_display = detailed.copy()
                for col in detailed_display.columns[1:]:
                    detailed_display[col] = detailed_display[col].apply(money)

                # Cabeçalhos abreviados para caberem
                header_mapping = {
                    "Modelo": "Modelo",
                    "Custo total anual": "Custo total\nanual",
                    "Custo mensal médio": "Custo mensal\nmédio",
                    "Combustível/energia": "Comb./\nenerg.",
                    "IPVA": "IPVA",
                    "Licenciamento": "Licenciam.",
                    "Seguro": "Seguro",
                    "Revisões (média anual)": "Revisões",
                    "Manutenção": "Manutenção",
                    "Pneus": "Pneus",
                    "Estacionamento": "Estacion.",
                    "Outros": "Outros",
                    "Assinatura": "Assinatura",
                    "Custo de oportunidade": "Custo\nde oport.",
                    "Juros financiamento": "Juros\nfinanc.",
                }
                headers = [header_mapping.get(col, col) for col in detailed_display.columns]
                detailed_data = [headers] + detailed_display.values.tolist()
                available_width = 18 * cm
                num_cols = len(detailed_display.columns)
                first_col_width = 4 * cm
                other_cols_width = (available_width - first_col_width) / (num_cols - 1)
                col_widths = [first_col_width] + [other_cols_width] * (num_cols - 1)
                detail_table = Table(detailed_data, repeatRows=1, colWidths=col_widths)
                # Estilo com cores para componentes e totalizadores
                # Aplicar cor de fundo nas células de dados conforme o tipo de coluna
                total_indices = [i for i, col in enumerate(detailed_display.columns) if col in total_cols]
                component_indices = [i for i, col in enumerate(detailed_display.columns) if col in component_cols]
                style_cmds = [
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('FONTSIZE', (0,0), (-1,0), 5),
                    ('FONTSIZE', (0,1), (-1,-1), 5),
                    ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('LEFTPADDING', (0,0), (-1,-1), 1),
                    ('RIGHTPADDING', (0,0), (-1,-1), 1),
                    ('TOPPADDING', (0,0), (-1,-1), 1),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ]
                # Aplicar cor verde para totalizadores, azul para componentes
                for row_idx in range(1, len(detailed_data)):
                    for col_idx in total_indices:
                        style_cmds.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#c6efce')))
                    for col_idx in component_indices:
                        style_cmds.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#bdd7ee')))
                detail_table.setStyle(TableStyle(style_cmds))
                story.append(detail_table)
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph(
                    "🟦 Componentes do custo 🟩 Totalizadores (Custo total anual e Custo mensal médio). "
                    "O <b>Custo mensal médio</b> é o Custo total anual médio dividido por 12, "
                    "representando a média mensal de todos os custos ao longo do horizonte.",
                    styles["Normal"]
                ))
                story.append(PageBreak())

                story.append(Paragraph("Gráficos da Análise", heading_style))
                if img1:
                    story.append(Paragraph("Custo Anual Médio por Modelo", styles["Heading3"]))
                    story.append(Image(io.BytesIO(img1), width=16*cm, height=9*cm))
                    story.append(Spacer(1, 0.5*cm))
                if img2:
                    story.append(Paragraph("Composição Média Anual dos Custos", styles["Heading3"]))
                    story.append(Image(io.BytesIO(img2), width=18*cm, height=10*cm))
                    story.append(Spacer(1, 0.5*cm))
                if img3:
                    story.append(Paragraph("Depreciação", styles["Heading3"]))
                    story.append(Image(io.BytesIO(img3), width=16*cm, height=9*cm))
                    story.append(Spacer(1, 0.5*cm))

                if be_images:
                    story.append(PageBreak())
                    story.append(Paragraph(f"Análise de Break‑even - {winner_name} como referência", heading_style))
                    for name, be_img in be_images:
                        if be_img:
                            story.append(Paragraph(f"<b>{name} vs {winner_name}</b>", styles["Heading3"]))
                            story.append(Image(io.BytesIO(be_img), width=16*cm, height=10*cm))
                            story.append(Spacer(1, 0.5*cm))

                doc.build(story)
                pdf_bytes = pdf_buffer.getvalue()
                st.download_button(
                    label="⬇️ Baixar Relatório PDF",
                    data=pdf_bytes,
                    file_name="comparacao_veiculos.pdf",
                    mime="application/pdf"
                )
                st.success("Relatório gerado com sucesso!")
    else:
        st.info("Instale as dependências acima para habilitar a geração de PDF.")

if __name__ == "__main__":
    main()
