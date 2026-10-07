FROM node:lts-slim

RUN apt update \
  && apt install -y --no-install-recommends git ca-certificates bash \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/server
COPY . .

RUN chown -R node:node /opt/server

USER node

RUN npm install

EXPOSE 8888/tcp
ENTRYPOINT ["/opt/server/start.sh"]
