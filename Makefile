.PHONY: bootstrap app api eval test clean

bootstrap:
	python scripts/bootstrap.py

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload --port 8000

eval:
	python scripts/evaluate.py

test:
	pytest -q

clean:
	rm -rf data/bronze data/silver data/gold data/index mlruns .pytest_cache
