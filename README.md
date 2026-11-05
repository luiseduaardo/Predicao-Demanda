# Predição de demanda

Projeto do desafio de previsão de demanda proposto pela Praso, plataforma de compras usada por restaurantes com entrega em até 24 horas. O objetivo é estimar a demanda semanal de cada produto a partir do histórico de pedidos e do índice de preço.

## Escopo

O trabalho tem quatro partes:

- Análise exploratória dos dados de pedidos e produtos, agregados por semana.
- Estimativa da elasticidade-preço da demanda por produto, por curva e por categoria.
- Modelo de previsão da demanda semanal de cada produto.
- Discussão sobre como usar o modelo em uma política de abastecimento.

## Dados

São duas tabelas em granularidade diária:

- Produtos: identificador, curva, categoria, disponibilidade em estoque, índice de preço e data de coleta.
- Pedidos: identificador do produto, quantidade e data do pedido.

Os arquivos CSV não são versionados. Coloque-os em `data/raw/`.

## Estrutura do repositório

```
.
├── data/
│   ├── raw/          # CSVs originais (não versionados)
│   └── processed/    # dados tratados
├── notebooks/        # análises e modelos
├── SPECIFICATIONS.md # enunciado do desafio
├── requirements.txt
└── LICENSE
```

## Como rodar

Crie um ambiente virtual, instale as dependências e abra os notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Resultados

todo

## Licença

MIT. Veja [LICENSE](LICENSE).
