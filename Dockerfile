FROM public.ecr.aws/lambda/python:3.13

COPY .. ${LAMBDA_TASK_ROOT}

RUN pip3 install -r requirements.txt
RUN pip3 install python-weather
LABEL authors="Nathaniel Rogalskyj"

CMD [ "deployment.spectral_api.parse_lambda_event" ]

#Local tag is provided by spectral_api
#docker build -t spectral_api .
#docker tag <local-tag>:latest <aws_account_id>.dkr.ecr.<region><repo-name>:latest
#docker tag spectral_api:latest 677276077251.dkr.ecr.us-east-2.amazonaws.com
#docker push .dkr.ecr.<region><repo-name>:latest
