FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir flask gunicorn pycryptodome PyPDF2 python-docx Pillow

COPY . .

CMD ["gunicorn", "app:app"]
