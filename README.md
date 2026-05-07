# Comparador de custo efetivo de carros

Aplicativo Streamlit para comparar custo total de propriedade entre carros no Brasil.

## O que considera

- Valor de compra, entrada, financiamento, juros e prazo.
- Combustivel ou energia, incluindo hibridos com percentual de uso eletrico.
- IPVA por estado como default editavel.
- Licenciamento, seguro, revisoes, manutencao, pneus, estacionamento e outros custos.
- Depreciacao e valor de revenda.
- Curva de ponto de virada contra o carro atual.

## Como rodar

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Os defaults de IPVA sao apenas pontos de partida. Confira regras atualizadas do seu estado, especialmente para eletricos, hibridos, isencoes, faixa de valor e ano do veiculo.
