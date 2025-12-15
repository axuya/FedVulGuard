import fs from "fs";
import path from "path";
import { parse } from "@nomicfoundation/slang";

// 输入目录：你的 raw 合约目录
const RAW_DIR = "data/raw";
const OUT_DIR = "data/graphs_ast_slang";

if (!fs.existsSync(OUT_DIR)) {
    fs.mkdirSync(OUT_DIR, { recursive: true });
}

const chains = ["Arbitrum", "Avalanche", "BSC", "Ethereum", "Fantom", "Polygon"];

function walk(dir) {
    let results = [];
    let list = fs.readdirSync(dir);

    list.forEach((file) => {
        file = path.join(dir, file);

        let stat = fs.statSync(file);
        if (stat && stat.isDirectory()) {
            results = results.concat(walk(file));
        } else if (file.endsWith(".sol")) {
            results.push(file);
        }
    });

    return results;
}

function nodeToJson(node) {
    return {
        type: node.kind,
        text: node.text,
        children: node.children.map(nodeToJson)
    };
}

for (const chain of chains) {
    console.log(`\n[SLANG] Processing chain: ${chain}`);

    const chainDir = path.join(RAW_DIR, chain);
    const solFiles = walk(chainDir);

    const outFile = path.join(OUT_DIR, `${chain}.jsonl`);
    const fout = fs.createWriteStream(outFile, { flags: "w" });

    let count = 0;

    for (const filePath of solFiles) {
        const id = path.basename(filePath, ".sol");

        const code = fs.readFileSync(filePath, "utf8");

        try {
            const parsed = parse(code); // Slang 核心解析

            const ast = nodeToJson(parsed.tree.root);

            fout.write(
                JSON.stringify({
                    id: id,
                    chain: chain,
                    ast_nodes: ast,
                    ast_edges: [] // 如果需要 AST 边结构，我可以帮你生成
                }) + "\n"
            );

            count++;
        } catch (e) {
            console.log(`[WARN] Slang failed: ${filePath}`);
        }
    }

    fout.end();
    console.log(`[OK] Saved AST for ${chain}, count = ${count} → ${outFile}`);
}
