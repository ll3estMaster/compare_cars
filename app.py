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
import plotly.express as px

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


SIMULATIONS_DIR = Path("simulacoes")
SIMULATIONS_DIR.mkdir(exist_ok=True)


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
        # Garantir compatibilidade com simulações antigas (sem valor_carro_atual_trade_in)
        if "valor_carro_atual_trade_in" not in car_data:
            car_data["valor_carro_atual_trade_in"] = 0.0
        cars.append(CarInputs(**car_data))
    return data["ipva_default"], data["horizon"], data["investment_return"], cars


def get_simulations_metadata():
    """Retorna lista de simulações com timestamp, modelos e dados principais."""
    simulations = []
    for filepath in sorted(SIMULATIONS_DIR.glob("*.json"), key=lambda x: os.path.getmtime(x), reverse=True):
        if filepath.stem.endswith("_DRAFT") or filepath.stem.startswith("_AUTO_"):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            cars_data = data.get("cars", [])
            model_names = [car.get("nome", "?") for car in cars_data]
            
            # Calcular dados principais de cada modelo
            models_info = []
            for car in cars_data:
                current_value = car.get("valor_base", 0)
                if car.get("carro_atual"):
                    current_value = car.get("valor_base", 0)
                
                # Calcular custo mensal aproximado (simplificado)
                monthly_cost = (
                    (current_value * car.get("ipva_percentual", 0) / 100 / 12) +
                    (car.get("seguro_anual", 0) / 12) +
                    (car.get("licenciamento_anual", 0) / 12) +
                    (car.get("manutencao_anual", 0) / 12) +
                    (car.get("pneus_anual", 0) / 12) +
                    (car.get("estacionamento_anual", 0) / 12) +
                    (car.get("outros_anuais", 0) / 12)
                )
                
                # Se é assinatura, adicionar valor da assinatura
                if car.get("pagamento") == "Assinatura":
                    monthly_cost = car.get("assinatura_mensal", 0)
                
                # Se é financiado, calcular parcela
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
                    "nome": car.get("nome", "?"),
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
    return [f.stem for f in SIMULATIONS_DIR.glob("*.json") if not f.stem.endswith("_DRAFT")]


def format_simulation_title(models):
    """Formata título com todos os modelos sem abreviar."""
    return " vs ".join(models)


def save_draft(ipva_default, horizon, investment_return, cars):
    """Salva rascunho. Cria arquivo numerado apenas após 1 hora sem alterações."""
    try:
        # Sempre salva o DRAFT atual
        save_simulation("_DRAFT", ipva_default, horizon, investment_return, cars)
        
        # Verifica se deve criar um novo arquivo (a cada 1 hora)
        if "last_auto_save_time" not in st.session_state:
            st.session_state.last_auto_save_time = datetime.now()
        
        current_time = datetime.now()
        time_elapsed = (current_time - st.session_state.last_auto_save_time).total_seconds()
        
        # Se passaram 1 hora (3600 segundos), salva como arquivo numerado
        if time_elapsed >= 3600:
            # Encontra o próximo número disponível
            existing_autos = list(SIMULATIONS_DIR.glob("_AUTO_*.json"))
            auto_number = len(existing_autos) + 1
            
            # Gera nome com modelos para contexto
            model_names = [car.get("nome", "?") for car in [car.__dict__ if hasattr(car, '__dict__') else car for car in cars]]
            model_str = "_".join(model_names).replace(" ", "_")[:50]  # Limita a 50 chars
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


def simulate_car(car: CarInputs, current_value: float, investment_return: float) -> pd.DataFrame:
    # Usar trade_in específico do modelo se definido, caso contrário usar current_value
    trade_in_value = getattr(car, 'valor_carro_atual_trade_in', 0.0)
    trade_in_value = trade_in_value if trade_in_value > 0 else current_value
    
    if car.carro_atual:
        additional_capital = 0.0
    elif car.pagamento == "A vista":
        additional_capital = max(0.0, car.valor_base - trade_in_value)
    elif car.pagamento == "Financiado":
        additional_capital = max(0.0, car.entrada_extra - trade_in_value)
    else:
        additional_capital = 0.0

    rate = investment_return / 100
    accumulated = additional_capital

    schedule = financing_schedule_by_year(car)

    rows = []
    for year in range(1, car.horizonte_anos + 1):
        annual_opportunity = accumulated * rate
        accumulated *= (1 + rate)

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
            "Modelo": car.nome,
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


def add_general_assumptions(ipva_default_val=4.0, horizon_val=3, investment_return_val=10.0):
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
                                     st.session_state.ipva_default, 0.1)
    st.session_state.ipva_default = ipva_default

    horizon = col2.slider("⏱️ Período de Análise (anos)", 1, 10, horizon_val)
    investment_return = col3.number_input("💰 Retorno Esperado de Investimentos (% ao ano)", 0.0, 30.0,
                                          investment_return_val, 0.5,
                                          help="Rendimento que você obteria investindo o capital adicional. "
                                               "Usado para calcular o custo de oportunidade com juros compostos.")
    # Retornar também o estado
    return state, ipva_default, horizon, investment_return


def car_form(index: int, ipva_default: float, horizon: int, current_value: float | None = None,
             default_car: CarInputs | None = None, investment_return: float = 10.0) -> CarInputs:
    current = index == 0
    label = "Carro atual" if current else f"Modelo {index}"

    # defaults
    d_nome = default_car.nome if default_car else label
    d_tipo = default_car.tipo if default_car else "Combustao"
    d_pagamento = default_car.pagamento if default_car and not current else "A vista"
    d_valor_base = default_car.valor_base if default_car else (122_000.0 if current else 225_000.0)
    d_km = default_car.km_mes if default_car else 900.0
    d_ipva = default_car.ipva_percentual if default_car else ipva_default
    d_seguro = default_car.seguro_anual if default_car else (d_valor_base * 0.035)
    d_lic = default_car.licenciamento_anual if default_car else 200.0
    d_manut = default_car.manutencao_anual if default_car else 1000.0
    d_pneus = default_car.pneus_anual if default_car else 0.0
    d_estac = default_car.estacionamento_anual if default_car else 0.0
    d_outros = default_car.outros_anuais if default_car else 0.0
    d_revisoes = list(default_car.revisoes_anuais) if default_car else [1200.0 + 250.0 * ((y - 1) % 3) for y in range(1, horizon + 1)]
    d_cons_l = default_car.consumo_km_l if default_car else (6.5 if d_tipo == "Combustao" else 15.0)
    d_cons_kwh = default_car.consumo_km_kwh if default_car else 6.0
    d_preco_comb = default_car.preco_combustivel if default_car else 6.50
    d_preco_kwh = default_car.preco_kwh if default_car else 1.30
    d_recarga = default_car.percentual_recarga_externa if default_car else 0.0
    d_entrada = default_car.entrada_extra if default_car else 0.0
    d_taxa = default_car.taxa_juros_am if default_car else 1.3
    d_prazo = default_car.prazo_meses if default_car else 24
    d_assinatura_m = default_car.assinatura_mensal if default_car else 3500.0
    d_assinatura_tx = default_car.taxa_inicial_assinatura if default_car else 0.0
    d_depr = default_car.depreciation_factor if default_car else 15.0
    d_trade_in = getattr(default_car, 'valor_carro_atual_trade_in', 0.0) if default_car else (current_value if current_value else 0.0)

    with st.expander(label, expanded=index <= 2):
        col1, col2, col3 = st.columns(3)

        with col1:
            nome = st.text_input("🚗 Nome/Modelo", d_nome, key=f"name_{index}")
            
            # Trade-in input apenas para modelos alternativos
            trade_in = 0.0
            if not current:
                trade_in = st.number_input(
                    "💰 Valor de troca do seu carro atual (proposta)",
                    0.0, 2_000_000.0, d_trade_in, 1000.0, key=f"trade_in_{index}")
                if current_value:
                    st.caption(f"Padrão (seu carro): {money(current_value)}")
            
            if not current:
                pagamento = st.selectbox("💳 Forma de Pagamento", ["A vista", "Financiado", "Assinatura"],
                                         index=["A vista", "Financiado", "Assinatura"].index(d_pagamento),
                                         key=f"pay_{index}")
            else:
                pagamento = d_pagamento
            tipo = st.selectbox("⚡ Tipo", ["Combustao", "Hibrido", "Eletrico"],
                                index=["Combustao", "Hibrido", "Eletrico"].index(d_tipo), key=f"type_{index}")
            valor_base = st.number_input(
                "💵 Valor de Mercado Atual" if current else ("💵 Preço do Modelo" if pagamento != "Assinatura" else "💵 Valor de Referência"),
                0.0, 2_000_000.0, d_valor_base, 1000.0, key=f"value_{index}")
            km_mes = st.number_input("📍 Quilometragem Mensal (km)", 0.0, 20_000.0, d_km, 100.0, key=f"km_{index}")
            ipva = d_ipva
            if pagamento != "Assinatura":
                ipva = st.number_input("📋 IPVA (% ao ano)", 0.0, 10.0, d_ipva, 0.1, key=f"ipva_{index}")

        with col2:
            # Initialize all consumption variables to avoid NameError
            consumo_l = d_cons_l
            consumo_kwh = d_cons_kwh
            preco_combustivel = d_preco_comb
            preco_kwh = d_preco_kwh
            percentual_recarga_externa = d_recarga

            if tipo == "Combustao":
                consumo_l = st.number_input("⛽ Consumo (km/l)", 0.1, 100.0, d_cons_l, 0.5, key=f"km_l_{index}")
                preco_combustivel = st.number_input("💰 Gasolina (R$/l)", 0.0, 20.0, d_preco_comb, 0.05, key=f"fuel_{index}")
            elif tipo == "Hibrido":
                consumo_l = st.number_input("⛽ Consumo Motor (km/l equiv.)", 0.1, 100.0, d_cons_l, 0.5, key=f"km_l_{index}")
                preco_combustivel = st.number_input("💰 Gasolina (R$/l)", 0.0, 20.0, d_preco_comb, 0.05, key=f"fuel_{index}")
                percentual_recarga_externa = st.slider("🔌 Recarga externa (%)", 0.0, 100.0, d_recarga, 5.0, key=f"ext_{index}")

            if tipo == "Eletrico" or (tipo == "Hibrido" and percentual_recarga_externa > 0):
                consumo_kwh = st.number_input("⚡ Eficiência (km/kWh)", 0.1, 20.0, d_cons_kwh, 0.1, key=f"km_kwh_{index}")
                preco_kwh = st.number_input("💰 Tarifa energia (R$/kWh)", 0.0, 5.0, d_preco_kwh, 0.05, key=f"kwh_{index}")

        with col3:
            entrada_extra = 0.0
            taxa = d_taxa
            prazo = d_prazo
            assinatura_mensal = d_assinatura_m
            taxa_inicial_assinatura = d_assinatura_tx

            if pagamento == "Financiado":
                pct = (d_entrada / valor_base * 100) if valor_base > 0 else 20.0
                pct = st.slider("💵 Entrada (%)", 0.0, 100.0, pct, 5.0, key=f"down_pct_{index}")
                entrada_extra = valor_base * pct / 100
                taxa = st.number_input("📈 Juros mensais (%)", 0.0, 10.0, d_taxa, 0.05, key=f"rate_{index}")
                prazo = st.number_input("⏱️ Prazo (meses)", 1, 120, d_prazo, 6, key=f"term_{index}")
                parcela = financing_monthly_payment(
                    CarInputs(id="", nome="", carro_atual=False, tipo="Combustao", valor_base=valor_base,
                              pagamento="Financiado", entrada_extra=entrada_extra, taxa_juros_am=taxa, prazo_meses=int(prazo),
                              assinatura_mensal=0, taxa_inicial_assinatura=0, km_mes=0, consumo_km_l=0, consumo_km_kwh=0,
                              preco_combustivel=0, preco_kwh=0, percentual_recarga_externa=0, ipva_percentual=0,
                              licenciamento_anual=0, seguro_anual=0, revisoes_anuais=(), manutencao_anual=0,
                              pneus_anual=0, estacionamento_anual=0, outros_anuais=0, horizonte_anos=horizon,
                              depreciation_factor=d_depr))
                st.success(f"💳 Parcela: {monthly_money(parcela)}")

                # --- NEW: Show detailed financing breakdown ---
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

            depr = d_depr
            if pagamento != "Assinatura":
                st.caption("📉 Depreciação (informativo)")
                depr = st.slider("Fator anual (%)", 0.0, 30.0, d_depr, 1.0, key=f"depr_{index}")

            if pagamento == "Assinatura":
                assinatura_mensal = st.number_input("📅 Mensalidade (R$/mês)", 0.0, 100_000.0, d_assinatura_m, 100.0, key=f"sub_m_{index}")
                taxa_inicial_assinatura = st.number_input("💳 Taxa inicial (R$)", 0.0, 100_000.0, d_assinatura_tx, 500.0, key=f"sub_fee_{index}")
                seguro = 0.0
                licenciamento = 0.0
                st.info("IPVA, seguro, licenciamento e manutenção inclusos na assinatura.")
            else:
                seguro = st.number_input("🛡️ Seguro anual (R$)", 0.0, 100_000.0, d_seguro, 250.0, key=f"ins_{index}")
                licenciamento = st.number_input("📝 Licenciamento anual (R$)", 0.0, 10_000.0, d_lic, 50.0, key=f"lic_{index}")

        # Maintenance block
        if pagamento == "Assinatura":
            st.caption("🚗 Custos adicionais")
            c1, c2 = st.columns(2)
            manutencao = 0.0
            pneus = 0.0
            estacionamento = c1.number_input("🅿️ Estacionamento/ano", 0.0, 100_000.0, d_estac, 100.0, key=f"park_{index}")
            outros = c2.number_input("📦 Outros/ano", 0.0, 100_000.0, d_outros, 100.0, key=f"other_{index}")
            revisoes = tuple(0.0 for _ in range(horizon))
        else:
            st.caption("🔧 Manutenção")
            c1, c2, c3, c4 = st.columns(4)
            manutencao = c1.number_input("🔧 Manutenção/ano", 0.0, 100_000.0, d_manut, 100.0, key=f"maint_{index}")
            pneus = c2.number_input("⛩️ Pneus/ano", 0.0, 100_000.0, d_pneus, 100.0, key=f"tires_{index}")
            estacionamento = c3.number_input("🅿️ Estacionamento/ano", 0.0, 100_000.0, d_estac, 100.0, key=f"park_{index}")
            outros = c4.number_input("📦 Outros/ano", 0.0, 100_000.0, d_outros, 100.0, key=f"other_{index}")

            st.caption("💰 Revisões")
            rev_cols = st.columns(min(horizon, 5))
            revisoes = []
            for y in range(1, horizon + 1):
                col = rev_cols[(y - 1) % len(rev_cols)]
                revisoes.append(col.number_input(f"Ano {y}", 0.0, 100_000.0,
                                                 d_revisoes[y - 1] if y <= len(d_revisoes) else 1200.0 + 250.0 * ((y - 1) % 3),
                                                 100.0, key=f"rev_{index}_{y}"))

            # Mostrar média das revisões
            media_revisoes = sum(revisoes) / len(revisoes) if revisoes else 0.0
            st.info(f"📊 **Média anual de revisões no período: {money(media_revisoes)}**")

        # Message about additional capital needed
        if not current:
            # Usar trade_in específico do modelo ao invés de current_value
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


def display_recent_simulations():
    """Exibe simulações recentes em lista vertical com opção de carregar."""
    simulations = get_simulations_metadata()
    
    if not simulations:
        return None
    
    st.subheader("⏱️ Últimas Simulações")
    
    # Mostrar as últimas 15 simulações em lista vertical
    recent = simulations[:15]
    
    for sim in recent:
        # Criar um expander para cada simulação
        models_display = format_simulation_title(sim['models'])
        
        with st.expander(f"📅 {sim['timestamp']} • {models_display}", expanded=False):
            # Botão para carregar
            if st.button(
                "📂 Carregar Simulação",
                key=f"load_recent_{sim['filename']}",
                use_container_width=True
            ):
                try:
                    ipva_default, horizon, investment_return, cars_temp = load_simulation(sim['filename'])
                    st.session_state.ipva_default = ipva_default
                    st.session_state.horizon = horizon
                    st.session_state.investment_return = investment_return
                    st.session_state.cars = cars_temp
                    st.session_state.draft_restored = True
                    st.success(f"✅ Simulação carregada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
    
    st.divider()
    return True


def make_break_even(current: CarInputs, target: CarInputs, investment_return: float) -> pd.DataFrame:
    """Gera malha de preços e consumos para calcular diferença no custo anual médio."""
    target_prices = np.linspace(max(target.valor_base * 0.6, 10_000), target.valor_base * 1.4, 30)
    if target.tipo == "Eletrico":
        consumptions = np.linspace(3.0, 10.0, 30)
    else:
        consumptions = np.linspace(5, 25, 30)

    # custo anual médio do carro atual
    current_annual = simulate_car(current, current.valor_base, investment_return)["Custo total anual"].mean()
    rows = []

    for price in target_prices:
        for consumption in consumptions:
            candidate = CarInputs(**target.__dict__)
            candidate.valor_base = float(price)
            # ajusta seguro proporcionalmente ao preço
            candidate.seguro_anual = price * (target.seguro_anual / max(target.valor_base, 1))
            if candidate.tipo == "Eletrico":
                candidate.consumo_km_kwh = float(consumption)
            else:
                candidate.consumo_km_l = float(consumption)
            target_annual = simulate_car(candidate, current.valor_base, investment_return)["Custo total anual"].mean()
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

    # Restore draft
    draft = load_draft()
    if draft and not st.session_state.get("draft_restored"):
        ipva_d, h, ir, cars_d = draft
        st.session_state.ipva_default = ipva_d
        st.session_state.horizon = h
        st.session_state.investment_return = ir
        st.session_state.cars = cars_d
        st.session_state.draft_restored = True
        st.info("✅ Rascunho restaurado.")

    # Exibir simulações recentes
    display_recent_simulations()

    ipva_default_val = st.session_state.get("ipva_default", 4.0)
    horizon_val = st.session_state.get("horizon", 3)
    investment_return_val = st.session_state.get("investment_return", 10.0)
    cars_loaded = st.session_state.get("cars", [])

    state, ipva_default, horizon, investment_return = add_general_assumptions(ipva_default_val, horizon_val, investment_return_val)

    # Load simulation
    with st.expander("📂 Carregar Simulação Salva", expanded=False):
        sims = list_simulations()
        if sims:
            sel = st.selectbox("Selecionar simulação", [""] + sims, key="load_sim")
            if st.button("✅ Carregar", key="btn_load") and sel:
                try:
                    ipva_default, horizon, investment_return, cars_temp = load_simulation(sel)
                    st.session_state.ipva_default = ipva_default
                    st.session_state.horizon = horizon
                    st.session_state.investment_return = investment_return
                    st.session_state.cars = cars_temp
                    st.success(f"Simulação '{sel}' carregada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao carregar: {e}")
        else:
            st.info("Nenhuma simulação salva.")

    st.subheader("🧠 Premissas da Análise")
    st.write(
        "**Custo de oportunidade**: incide apenas sobre o **capital adicional** que você precisa desembolsar **além do valor do seu carro atual**. "
        "Esse capital adicional deixa de ser investido – portanto o custo anual considera **juros compostos** (o rendimento perdido cresce a cada ano). "
        "Se o carro atual cobre a entrada (financiamento) ou o preço total (à vista), o custo de oportunidade é zero. "
        "O valor financiado não gera custo de oportunidade adicional."
    )

    current = car_form(0, ipva_default, horizon, default_car=cars_loaded[0] if cars_loaded else None,
                       investment_return=investment_return)
    n_models = st.number_input("💶 Quantos modelos alternativos comparar?", 1, 20,
                               value=len(cars_loaded) - 1 if cars_loaded else 2, step=1)
    cars = [current] + [car_form(i, ipva_default, horizon, current.valor_base,
                                 default_car=cars_loaded[i] if cars_loaded and i < len(cars_loaded) else None,
                                 investment_return=investment_return)
                        for i in range(1, int(n_models) + 1)]

    save_draft(ipva_default, horizon, investment_return, cars)

    # Save / manage with auto-generated name
    with st.expander("💾 Salvar / Gerenciar Simulações", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            # Generate default name from model names
            model_names = [car.nome for car in cars if car.nome.strip()]
            default_name = "_".join(model_names).replace(" ", "_")
            # Use session state to keep the name if user already typed something
            if "save_name_input" not in st.session_state:
                st.session_state.save_name_input = default_name
            # Button to regenerate name from current models
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

    # Simulation
    sim_df = pd.concat([simulate_car(car, current.valor_base, investment_return) for car in cars], ignore_index=True)
    annual_avg = sim_df.groupby(["Modelo", "ID"])["Custo total anual"].mean().reset_index()
    annual_avg["Custo mensal médio"] = annual_avg["Custo total anual"] / 12
    annual_avg = annual_avg.sort_values("Custo total anual")
    winner = annual_avg.iloc[0]

    st.subheader("🎯 Melhor Opção")
    st.metric(label=f"✅ {winner['Modelo']}", value=money(winner["Custo total anual"]), help="Custo anual médio")
    st.caption("Custo anual médio considerado ao longo do período da análise (já com juros compostos no custo de oportunidade).")

    # Comparison table
    st.dataframe(
        annual_avg[["Modelo", "Custo total anual", "Custo mensal médio"]]
        .style.format({"Custo total anual": money, "Custo mensal médio": money}),
        use_container_width=True,
    )

    # Opportunity cost detail
    st.subheader("💼 Custo de Oportunidade Detalhado")
    opp_data = []
    for car in cars:
        if car.carro_atual:
            cap_adicional = 0.0
            trade_in_value = 0.0
        else:
            # Usar trade_in específico do modelo se definido, caso contrário usar valor do carro atual
            trade_in_value = getattr(car, 'valor_carro_atual_trade_in', 0.0)
            trade_in_value = trade_in_value if trade_in_value > 0 else current.valor_base
            if car.pagamento == "A vista":
                cap_adicional = max(0.0, car.valor_base - trade_in_value)
            elif car.pagamento == "Financiado":
                cap_adicional = max(0.0, car.entrada_extra - trade_in_value)
            else:
                cap_adicional = 0.0
        first_year_opp = cap_adicional * investment_return / 100
        opp_data.append({
            "Modelo": car.nome,
            "Valor de Troca": trade_in_value,
            "Capital Adicional": cap_adicional,
            "Taxa de Retorno": f"{investment_return:.1f}%",
            "Custo Oport. (1º ano)": first_year_opp,
        })
    opp_df = pd.DataFrame(opp_data)
    st.dataframe(
        opp_df.style.format({"Capital Adicional": money, "Custo Oport. (1º ano)": money}),
        use_container_width=True,
    )
    st.caption("O custo de oportunidade real cresce ano a ano (juros compostos). "
               "A tabela abaixo mostra a média anual de cada categoria já considerando essa evolução.")

    # Detailed average annual cost per model (all in one table)
    st.subheader("📊 Detalhamento do Custo Anual Médio por Modelo")
    categories = ["Combustível/energia", "IPVA", "Licenciamento", "Seguro",
                  "Revisões (média anual)", "Manutenção", "Pneus", "Estacionamento", "Outros",
                  "Assinatura", "Custo de oportunidade", "Juros financiamento"]
    breakd = sim_df.groupby("Modelo")[["Combustível/energia", "IPVA", "Licenciamento", "Seguro",
                                       "Revisões", "Manutenção", "Pneus", "Estacionamento", "Outros",
                                       "Assinatura", "Custo de oportunidade", "Juros financiamento"]].mean().reset_index()
    breakd = breakd.rename(columns={"Revisões": "Revisões (média anual)"})
    detailed = annual_avg[["Modelo", "Custo total anual", "Custo mensal médio"]].merge(breakd, on="Modelo")
    col_order = ["Modelo", "Custo total anual", "Custo mensal médio"] + categories
    detailed = detailed[col_order]
    st.dataframe(
        detailed.style.format({col: money for col in col_order if col != "Modelo"}),
        use_container_width=True,
    )

    # ---------- Gráficos ----------
    st.subheader("📈 Gráficos")

    # Paleta com 24 cores vibrantes (suficiente para até 24 categorias)
    custom_palette = px.colors.qualitative.Plotly

    # 1. Custo anual médio por modelo
    fig1 = px.bar(annual_avg, x="Modelo", y="Custo total anual",
                text=annual_avg["Custo total anual"].map(money),
                color="Modelo", title="Custo Anual Médio por Modelo",
                color_discrete_sequence=custom_palette)
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=450, width=800)
    st.plotly_chart(fig1, use_container_width=False, width=800)

    # 2. Composição Média Anual (barras empilhadas)
    bar_data = breakd.melt("Modelo", var_name="Categoria", value_name="Custo")
    bar_data["Rótulo"] = bar_data.apply(lambda r: f"{r['Categoria']}: {short_money(r['Custo'])}", axis=1)
    fig2 = px.bar(bar_data, x="Modelo", y="Custo", color="Categoria", text="Rótulo",
                title="Composição Média Anual dos Custos",
                color_discrete_sequence=custom_palette)
    fig2.update_traces(textposition="inside", insidetextanchor="middle",
                    textfont=dict(size=10, color="black", family="sans-serif"))
    fig2.update_layout(height=500, width=900,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=False, width=900)

    # 3. Depreciação (linhas)
    dep_data = []
    for car in cars:
        if car.pagamento != "Assinatura":
            rate = car.depreciation_factor / 100
            val = car.valor_base if not car.carro_atual else current.valor_base
            for y in range(1, horizon + 1):
                dep_data.append({"Modelo": car.nome, "Ano": y, "Valor": val * (1 - rate) ** y})
    if dep_data:
        dep_df = pd.DataFrame(dep_data)
        dep_df["Rotulo"] = dep_df["Valor"].map(short_money)
        fig3 = px.line(dep_df, x="Ano", y="Valor", color="Modelo", markers=True, text="Rotulo",
                    title="Evolução do valor do veículo",
                    color_discrete_sequence=custom_palette)
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=450, width=800)
        st.plotly_chart(fig3, use_container_width=False, width=800)

    # 4. Break‑even
    st.subheader("📈 Ponto de Virada (Break-Even)")
    st.markdown("Explore como preço e consumo afetariam a decisão. Valores negativos (azul) = modelo alternativo mais barato que o atual.")
    target_options = {f"{car.nome} ({car.id})": car for car in cars if car.id != current.id}
    if target_options:
        target_label = st.selectbox("🔍 Modelo para Testar contra o Carro Atual", list(target_options.keys()), key="be_select")
        target = target_options[target_label]
        with st.spinner("Gerando malha de break‑even..."):
            be_df = make_break_even(current, target, investment_return)
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
            name=f"{target.nome} atual",
        ))
        fig_be.update_layout(
            title=f"Break‑even: {target.nome} vs {current.nome}",
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

    # ================== BOTÃO PDF (SOMENTE UM, CORRIGIDO) ==================
    st.subheader("📄 Relatório em PDF")
    
    # Verificar disponibilidade das bibliotecas
    try:
        import reportlab
        import kaleido
        from reportlab.lib.pagesizes import A4, landscape
        pdf_available = True
    except ImportError as e:
        pdf_available = False
        missing = str(e).split("'")[1] if "'" in str(e) else "reportlab ou kaleido"
        st.warning(f"⚠️ Para gerar o PDF, instale as bibliotecas necessárias:\n\n```\npip install reportlab kaleido\n```\n\nFaltando: `{missing}`")

    if pdf_available:
        if st.button("📄 Gerar Relatório PDF", key="btn_pdf"):
            # Imports necessários (feitos aqui para não pesar o carregamento inicial)
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import cm
            import plotly.io as pio
            import io
            import tempfile

            with st.spinner("Gerando relatório PDF... Isso pode levar alguns segundos."):
                # Determinar modelo vencedor
                winner_name = annual_avg.iloc[0]["Modelo"]
                winner_car = next(car for car in cars if car.nome == winner_name)
                other_cars = [car for car in cars if car.nome != winner_name]

                # Reconstruir figuras para o PDF (garantia)
                custom_palette = px.colors.qualitative.Plotly
                fig1_pdf = px.bar(annual_avg, x="Modelo", y="Custo total anual",
                                 text=annual_avg["Custo total anual"].map(money),
                                 color="Modelo", title="Custo Anual Médio por Modelo",
                                 color_discrete_sequence=custom_palette)
                fig1_pdf.update_traces(textposition="outside")
                fig1_pdf.update_layout(height=450, width=800)

                bar_data_pdf = breakd.melt("Modelo", var_name="Categoria", value_name="Custo")
                bar_data_pdf["Rótulo"] = bar_data_pdf.apply(lambda r: f"{r['Categoria']}: {short_money(r['Custo'])}", axis=1)
                fig2_pdf = px.bar(bar_data_pdf, x="Modelo", y="Custo", color="Categoria", text="Rótulo",
                                 title="Composição Média Anual dos Custos",
                                 color_discrete_sequence=custom_palette)
                fig2_pdf.update_traces(textposition="inside", insidetextanchor="middle",
                                      textfont=dict(size=10, color="black"))
                fig2_pdf.update_layout(height=500, width=900)

                dep_data_pdf = []
                for car in cars:
                    if car.pagamento != "Assinatura":
                        rate = car.depreciation_factor / 100
                        val = car.valor_base if not car.carro_atual else current.valor_base
                        for y in range(1, horizon + 1):
                            dep_data_pdf.append({"Modelo": car.nome, "Ano": y, "Valor": val * (1 - rate) ** y})
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
                        st.error(f"Erro ao converter gráfico para imagem: {e}")
                        return None

                img1 = fig_to_image(fig1_pdf)
                img2 = fig_to_image(fig2_pdf)
                img3 = fig_to_image(fig3_pdf) if fig3_pdf else None

                # Break-even plots
                be_images = []
                for other in other_cars:
                    be_df = make_break_even(winner_car, other, investment_return)
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
                        name=f"{other.nome} atual",
                    ))
                    fig_be.update_layout(
                        title=f"Break‑even: {other.nome} vs {winner_car.nome}",
                        xaxis_title="Preço do veículo (R$)",
                        yaxis_title=f"Consumo ({unit})",
                        height=500,
                        width=800,
                    )
                    be_img = fig_to_image(fig_be, width=800)
                    if be_img:
                        be_images.append((other.nome, be_img))

                # Construir PDF
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
                styles = getSampleStyleSheet()
                title_style = styles["Title"]
                heading_style = styles["Heading2"]
                normal_style = styles["Normal"]

                story = []

                # Título
                story.append(Paragraph("Relatório de Comparação de Custos de Veículos", title_style))
                story.append(Spacer(1, 0.5*cm))

                # Premissas
                story.append(Paragraph("Premissas Gerais", heading_style))
                story.append(Paragraph(f"Estado para IPVA padrão: {state} (alíquota {ipva_default:.1f}%)", normal_style))
                story.append(Paragraph(f"Período de análise: {horizon} anos", normal_style))
                story.append(Paragraph(f"Retorno esperado de investimentos: {investment_return:.1f}% a.a.", normal_style))
                story.append(Spacer(1, 0.5*cm))

                # Tabela de modelos comparados
                story.append(Paragraph("Modelos Analisados", heading_style))
                model_table_data = [["Modelo", "Tipo", "Pagamento", "Valor (R$)", "Km/mês"]]
                for car in cars:
                    model_table_data.append([
                        car.nome,
                        car.tipo,
                        car.pagamento,
                        money(car.valor_base),
                        f"{car.km_mes:.0f}"
                    ])
                model_table = Table(model_table_data, colWidths=[4*cm, 3*cm, 3.5*cm, 3.5*cm, 3*cm])
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

                # Inputs detalhados por carro (incluindo valor de troca do carro atual)
                story.append(Paragraph("Dados Completos de Entrada", heading_style))
                for car in cars:
                    story.append(Paragraph(f"<b>{car.nome}</b>", styles["Heading3"]))
                    params = [
                        ["Tipo", car.tipo],
                        ["Forma de pagamento", car.pagamento],
                        ["Valor base", money(car.valor_base)],
                        ["Km/mês", f"{car.km_mes:.0f}"],
                    ]
                    
                    # --- INFORMAÇÃO SOBRE VALOR DE TROCA (TRADE-IN) ---
                    if car.carro_atual:
                        # Para o carro atual, mostra seu valor de mercado (base para troca)
                        params.append(["Valor do seu carro atual", money(car.valor_base)])
                    else:
                        # Para modelo alternativo, mostra o valor oferecido pelo seu carro atual
                        trade_offer = getattr(car, 'valor_carro_atual_trade_in', 0.0)
                        if trade_offer == 0.0:
                            # Se não definido (simulação antiga), assume o valor do carro atual como padrão
                            trade_offer = current.valor_base
                            params.append(["Valor oferecido pelo seu carro", money(trade_offer)])
                        else:
                            params.append(["Valor oferecido pelo seu carro", money(trade_offer)])
                        
                        # Capital adicional necessário
                        if car.pagamento == "A vista":
                            cap_adicional = max(0.0, car.valor_base - trade_offer)
                        elif car.pagamento == "Financiado":
                            cap_adicional = max(0.0, car.entrada_extra - trade_offer)
                        else:
                            cap_adicional = 0.0
                        params.append(["Capital adicional necessário", money(cap_adicional) if cap_adicional > 0 else "Nenhum"])
                    
                    # IPVA, seguro, etc.
                    params.append(["IPVA (%)", f"{car.ipva_percentual:.1f}%" if car.pagamento != "Assinatura" else "Incluso"])
                    params.append(["Seguro anual", money(car.seguro_anual) if car.pagamento != "Assinatura" else "Incluso"])
                    params.append(["Licenciamento anual", money(car.licenciamento_anual) if car.pagamento != "Assinatura" else "Incluso"])
                    params.append(["Manutenção anual", money(car.manutencao_anual) if car.pagamento != "Assinatura" else "Incluso"])
                    params.append(["Pneus anual", money(car.pneus_anual) if car.pagamento != "Assinatura" else "Incluso"])
                    params.append(["Estacionamento anual", money(car.estacionamento_anual)])
                    params.append(["Outros anuais", money(car.outros_anuais)])
                    if car.pagamento != "Assinatura":
                        media_rev = sum(car.revisoes_anuais)/len(car.revisoes_anuais) if car.revisoes_anuais else 0
                        params.append(["Revisões médias anuais", money(media_rev)])
                    else:
                        params.append(["Revisões médias anuais", "Incluso"])

                    if car.tipo == "Combustao":
                        params.append(["Consumo", f"{car.consumo_km_l:.1f} km/l"])
                        params.append(["Preço combustível", money(car.preco_combustivel)+"/l"])
                    elif car.tipo == "Eletrico":
                        params.append(["Consumo", f"{car.consumo_km_kwh:.1f} km/kWh"])
                        params.append(["Preço energia", money(car.preco_kwh)+"/kWh"])
                    else:  # Hibrido
                        params.append(["Consumo gasolina", f"{car.consumo_km_l:.1f} km/l"])
                        params.append(["Preço gasolina", money(car.preco_combustivel)+"/l"])
                        params.append(["Recarga externa", f"{car.percentual_recarga_externa:.0f}%"])
                        params.append(["Consumo elétrico", f"{car.consumo_km_kwh:.1f} km/kWh"])
                        params.append(["Preço energia", money(car.preco_kwh)+"/kWh"])

                    if car.pagamento == "Financiado":
                        params.append(["Entrada extra", money(car.entrada_extra)])
                        params.append(["Taxa juros mensal", f"{car.taxa_juros_am:.2f}%"])
                        params.append(["Prazo (meses)", f"{car.prazo_meses}"])
                    elif car.pagamento == "Assinatura":
                        params.append(["Assinatura mensal", money(car.assinatura_mensal)])
                        params.append(["Taxa inicial", money(car.taxa_inicial_assinatura)])

                    if car.pagamento != "Assinatura":
                        params.append(["Depreciação anual", f"{car.depreciation_factor:.1f}%"])

                    # Montagem da tabela de parâmetros
                    param_table_data = [[params[i][0], params[i][1]] for i in range(len(params))]
                    param_table = Table(param_table_data, colWidths=[5*cm, 5*cm])
                    param_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ]))
                    story.append(param_table)
                    story.append(Spacer(1, 0.3*cm))
                story.append(PageBreak())

                # Resultados: tabelas
                story.append(Paragraph("Resultados da Simulação", heading_style))
                story.append(Paragraph("Custo Anual Médio e Mensal", heading_style))
                comp_data = [["Modelo", "Custo Anual Médio", "Custo Mensal Médio"]]
                for _, row in annual_avg.iterrows():
                    comp_data.append([row["Modelo"], money(row["Custo total anual"]), money(row["Custo mensal médio"])])
                comp_table = Table(comp_data, colWidths=[4*cm, 4*cm, 4*cm])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ]))
                story.append(comp_table)
                story.append(Spacer(1, 0.5*cm))

                story.append(Paragraph("Detalhamento dos Custos Médios Anuais", heading_style))

                # Prepara os dados da tabela
                detailed_display = detailed.copy()
                for col in detailed_display.columns[1:]:
                    detailed_display[col] = detailed_display[col].apply(money)

                # Quebrar cabeçalhos em múltiplas linhas manualmente
                header_mapping = {
                    "Modelo": "Modelo",
                    "Custo total anual": "Custo total\nanual",
                    "Custo mensal médio": "Custo mensal\nmédio",
                    "Combustível/energia": "Combustível/\nenergia",
                    "IPVA": "IPVA",
                    "Licenciamento": "Licenciamento",
                    "Seguro": "Seguro",
                    "Revisões (média anual)": "Revisões\n(média anual)",
                    "Manutenção": "Manutenção",
                    "Pneus": "Pneus",
                    "Estacionamento": "Estacionamento",
                    "Outros": "Outros",
                    "Assinatura": "Assinatura",
                    "Custo de oportunidade": "Custo de\noportunidade",
                    "Juros financiamento": "Juros\nfinanciamento",
                }

                headers = [header_mapping.get(col, col) for col in detailed_display.columns]
                detailed_data = [headers] + detailed_display.values.tolist()

                # Largura total útil em A4 retrato (21cm - margens de 1.5cm cada = 18cm)
                available_width = 18 * cm
                num_cols = len(detailed_display.columns)
                col_width = available_width / num_cols

                # Cria a tabela com larguras proporcionais
                detail_table = Table(detailed_data, repeatRows=1, colWidths=[col_width] * num_cols)

                # Estilo com fonte menor e centralizado
                detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('FONTSIZE', (0,0), (-1,0), 5),           # cabeçalho com fonte 5
                    ('FONTSIZE', (0,1), (-1,-1), 5),          # dados também 5
                    ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('LEFTPADDING', (0,0), (-1,-1), 1),
                    ('RIGHTPADDING', (0,0), (-1,-1), 1),
                    ('TOPPADDING', (0,0), (-1,-1), 1),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ]))

                story.append(detail_table)
                story.append(PageBreak())

                # Gráficos
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

                # Break-even contra o vencedor
                if be_images:
                    story.append(PageBreak())
                    story.append(Paragraph(f"Análise de Break‑even - {winner_car.nome} como referência", heading_style))
                    for name, be_img in be_images:
                        if be_img:
                            story.append(Paragraph(f"<b>{name} vs {winner_car.nome}</b>", styles["Heading3"]))
                            story.append(Image(io.BytesIO(be_img), width=16*cm, height=10*cm))
                            story.append(Spacer(1, 0.5*cm))

                # Finalizar PDF
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
