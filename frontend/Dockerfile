# syntax=docker/dockerfile:1
FROM node:14
WORKDIR /app
COPY . .
WORKDIR /app/cdfe
RUN npm install -g serve
RUN npm install
RUN npm run-script build
CMD serve -s build
