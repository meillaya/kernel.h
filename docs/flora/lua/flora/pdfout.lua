-- Writes a painter's output as a standalone one-page PDF, without TeX. The
-- page holds the painted content alone, with the opacity states and
-- gradient patterns it registered (paint.lua) as page resources.

local Paint = require("flora.paint")

local M = {}
local BP = 72 / 72.27

-- Painter `pt` as the bytes of a PDF page `width` x `height` pt (TeX
-- points).
function M.encode(pt, width, height)
  local res = Paint.resources()
  -- Objects: 1 catalog, 2 page tree, 3 page, 4 content, 5.. patterns.
  local objects = {}
  local gs, pats = {}, {}
  for _, g in ipairs(res.gs) do gs[#gs + 1] = string.format("/%s %s", g[1], g[2]) end
  for i, p in ipairs(res.patterns) do
    objects[4 + i] = p[2]
    pats[#pats + 1] = string.format("/%s %d 0 R", p[1], 4 + i)
  end
  local content = pt:body()
  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
  objects[2] = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
  objects[3] = string.format("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] "
    .. "/Resources << /ExtGState << %s >> /Pattern << %s >> >> /Contents 4 0 R >>",
    width * BP, height * BP, table.concat(gs, " "), table.concat(pats, " "))
  objects[4] = string.format("<< /Length %d >>\nstream\n%s\nendstream", #content, content)

  local parts, pos = {}, 0
  local function put(s) parts[#parts + 1] = s; pos = pos + #s end
  put("%PDF-1.5\n")
  local offsets = {}
  for i = 1, #objects do
    offsets[i] = pos
    put(string.format("%d 0 obj\n%s\nendobj\n", i, objects[i]))
  end
  local xref = pos
  put(string.format("xref\n0 %d\n0000000000 65535 f \n", #objects + 1))
  for i = 1, #objects do put(string.format("%010d 00000 n \n", offsets[i])) end
  put(string.format("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", #objects + 1, xref))
  return table.concat(parts)
end

-- Write painter `pt` to `path` as a PDF page `width` x `height` pt.
function M.write(path, pt, width, height)
  local fh = assert(io.open(path, "wb"))
  fh:write(M.encode(pt, width, height))
  fh:close()
end

return M
