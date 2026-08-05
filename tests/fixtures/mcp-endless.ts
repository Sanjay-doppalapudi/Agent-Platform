// A hostile MCP server for tests: tools/list ALWAYS returns a nextCursor.
// Mode is taken from argv[2]: "repeat" reuses one cursor, "fresh" invents a new
// one every page, "empty" returns fresh cursors with no tools at all.
const mode = process.argv[2] ?? "fresh";
let page = 0;

const respond = (id: number, result: unknown) =>
  process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id, result }) + "\n");

let buf = "";
process.stdin.on("data", (chunk: Buffer) => {
  buf += chunk.toString("utf8");
  let nl: number;
  while ((nl = buf.indexOf("\n")) !== -1) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let msg: any;
    try { msg = JSON.parse(line); } catch { continue; }
    if (msg.method === "initialize") {
      respond(msg.id, {
        protocolVersion: msg.params?.protocolVersion ?? "2025-06-18",
        capabilities: { tools: {} },
        serverInfo: { name: "endless", version: "1.0.0" },
      });
    } else if (msg.method === "tools/list") {
      page++;
      const tools = mode === "empty" ? [] : [{
        name: `tool_${page}`,
        description: "page tool",
        inputSchema: { type: "object", properties: {} },
      }];
      const id = msg.id;
      // Answer at a realistic pace. Replying instantly makes an unbounded
      // client loop flood this process to death, which masks the hang the
      // test is about (a well-behaved server just keeps answering).
      setTimeout(() => {
        respond(id, {
          tools,
          nextCursor: mode === "repeat" ? "SAME" : `cursor-${page}`, // never terminates
        });
      }, 4);
    } else if (msg.id != null) {
      respond(msg.id, {});
    }
  }
});
