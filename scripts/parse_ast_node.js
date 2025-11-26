const fs = require("fs");
const Parser = require("tree-sitter");
const Solidity = require("tree-sitter-solidity");

const code = fs.readFileSync(process.argv[2], "utf8");

const parser = new Parser();
parser.setLanguage(Solidity);

const tree = parser.parse(code);

// 输出 JSON 结构 AST（推荐）
function toJSON(node) {
  return {
    type: node.type,
    startPosition: node.startPosition,
    endPosition: node.endPosition,
    children: node.children.map(toJSON)
  };
}

console.log(JSON.stringify(toJSON(tree.rootNode), null, 2));
