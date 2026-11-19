# mitmproxy

[mitmproxy](https://github.com/mitmproxy/mitmproxy) is an intercepting proxy.
Similar tools are:

- Postman
- Fiddler Everywhere (or the classic Fiddler, if you still have it)

Local LLMs can be served without HTTPS, so it becomes easier to capture the
traffic without worrying about the certificates.

mitmproxy has the official Docker image. The following command starts a container:

- `:9932` serves as the reverse proxy. All traffic is forwarded to
  `http://llama.cpp:9931`. Replace that address with the actual IP and port.
- `:9933` provides the WebUI where the traffic can be inspected.

```bash
docker run --rm -it \
  --name mitmproxy \
  -p 9932:8080 \
  -p 127.0.0.1:9933:8081 \
  mitmproxy/mitmproxy \
  mitmweb \
  --mode reverse:http://llama.cpp:9931 \
  --listen-host 0.0.0.0 \
  --listen-port 8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --set web_password=localdebug123
```

> [!NOTE]
> This tool has nothing to do with a man-in-the-middle attack by default, unless
> you want to use it to capture HTTPS requests.

## Basic usage

The main part of the WebUI is split into two columns. The left column shows requests,
including the URL, method, status code, etc. The right column shows the details,
such as request, response, timing, etc.

The body of a request or response may contain long lines. By default, mitmproxy
doesn't wrap the text. A workaround is to manually add a new CSS rule using the
browser's inspector:

```css
.contentview pre {
    white-space: pre-wrap !important;
}
```

> [!TIP]
> Select a line in the body and look for the `.contentview pre` rule in your
> browser's inspector. Clicking a blank area inside the block may add a new
> rule.

## Filter

Switch to the `Flow List` tab at the top to find filters such as `Search` and
`Highlight`. They accept similar syntax. For example, `~b foo` filters flows
whose request or response body contains the string `foo`. `Search` only displays
the flows matching the rule, while `Highlight` displays all the flows, but
highlights the matching flows in the list.

Use a regular expression after `~b`. It can also take a longer sentence with double
quotes if it contains spaces. `~bq` searches only the request body, while `~bs`
searches only the response body. Focusing on the two inputs will pop up a table
of available commands and explanations.

## Export

You can also export the flows to [HAR format](https://en.wikipedia.org/wiki/HAR_(file_format)).
HAR is a de facto industry standard, so that later the records can be viewed in other
tools, such as:

- Chromium / Chrome
- Fiddler
- Firefox
- LoadRunner
- Postman

To export via the WebUI, first go to the `Options` tab to enable `Display Command
Bar`, then in the command bar, type the command `save.har @all /tmp/capture.har`.
Then copy the file from the Docker container to the host:

```bash
docker cp mitmproxy:/tmp/capture.har ./capture.har
```

> [!NOTE]
> Chrome may report errors when importing the HAR file. For instance, I got an
> error involving `NaN`. Ask a language model to inspect with `jq`.
