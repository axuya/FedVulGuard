const fs = require("fs");
const parser = require("solidity-parser-antlr");

function preprocess(code) {
    // 删除所有 inline assembly（它是 parser 的最大 bug 来源）
    code = code.replace(/assembly\s*{[^}]*}/gs, "assembly {}");

    // 删除使用模式错误
    code = code.replace(/using\s+[^;]+;/g, "");

    // 删除末尾非 UTF-8 字符
    code = code.replace(/[^\x00-\x7F]+$/g, "");

    return code;
}

const file = process.argv[2];
const code = preprocess(fs.readFileSync(file, "utf8"));

try {
    const ast = parser.parse(code, {
        tolerant: true,
        loc: false,
        range: false
    });
    console.log(JSON.stringify(ast));
} catch (e) {
    console.log(JSON.stringify({ error: e.toString() }));
    process.exit(0);   // 不抛错，正常退出
}
