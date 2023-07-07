# # 
FROM huggingface/transformers-cpu:latest
# # 
# WORKDIR /code


# # RUN apt-get update
# #

# COPY ./requirements.txt /code/requirements.txt

# # 
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# FROM nlp:latest

COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install -r /code/requirements.txt

# 
COPY ./ /code/

# 
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2468"]
CMD ["python3","api/main.py"]