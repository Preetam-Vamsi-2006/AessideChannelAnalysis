FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir flask gunicorn pycryptodome PyPDF2 python-docx Pillow

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
