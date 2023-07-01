format:
	black .
	isort .
test:
	pytest tests/
run:
	streamlit run app.py

