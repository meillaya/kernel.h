-- Document settings from \florasetup and \florathispage: a key=value list,
-- parsed and checked here so mistakes are reported with the key's name.
--
--   layout    left, right, centered, split, title or surround (frames.lua)
--   seed      a number, or random
--   mix       weights per kind of growth, e.g. mix={lily=2, umbel=0}
--             (mix.lua lists the kinds)
--   density   how full the border grows (mix.density); below 1 is airier
--   visitors  the weight of every visitor at once (bee, butterfly, bird,
--             snail); mix={bee=0} after it still turns one off
--   palette   what paints each part, e.g. palette={leaves=green,
--             hero=kon-peki}: look, an ink group, one ink (inks.lua), or
--             random for one ink picked per page
--   spin      a number that re-picks the random inks; the shapes stay

local M = {}

local Frames = require("flora.frames")
local Tune = require("flora.tune")
local Inks = require("flora.inks")
local Mix = require("flora.mix")

-- Key name -> function(value string) returning the parsed value, or nil
-- and a message.
local keys = {}

function M.define(name, parse)
  keys[name] = parse
end

M.define("layout", function(v)
  if Frames.layouts[v] then return v end
  return nil, "layout must be one of left, right, centered, split, title, surround"
end)

M.define("seed", function(v)
  if v == "random" or tonumber(v) then return v end
  return nil, "seed must be a number or random"
end)

local function weight(v)
  local n = tonumber(v)
  if n and n >= 0 and n <= 10 then return n end
end

local function known(t)
  local out = {}
  for k in pairs(t) do out[#out + 1] = k end
  table.sort(out)
  return table.concat(out, ", ")
end

M.define("density", function(v)
  local n = weight(v)
  if n then return n end
  return nil, "density must be a number from 0 to 10"
end)

-- A shorthand for every visitor's weight.
M.define("visitors", function(v)
  local n = weight(v)
  if n then return n end
  return nil, "visitors must be a number from 0 to 10"
end)

M.define("mix", function(v)
  local t = {}
  for _, kv in ipairs(M.parse_list(v)) do
    local k, w = kv[1], weight(kv[2])
    if Tune.mix[k] == nil then return nil, "no kind \"" .. k .. "\" (kinds: " .. known(Tune.mix) .. ")" end
    if not w then return nil, k .. " needs a weight from 0 to 10" end
    t[k] = w
  end
  return t
end)

M.define("palette", function(v)
  local t = {}
  for _, kv in ipairs(M.parse_list(v)) do
    local part, ink = kv[1], kv[2]
    if Tune.color.palette[part] == nil then
      return nil, "no part \"" .. part .. "\" (parts: " .. known(Tune.color.palette) .. ")"
    end
    if not (ink == "look" or ink == "random" or Inks.groups[ink] or Inks.ramps[ink]) then
      return nil, "\"" .. tostring(ink) .. "\" is not look, random, an ink group or an ink name"
    end
    t[part] = ink
  end
  return t
end)

M.define("spin", function(v)
  local n = tonumber(v)
  if n and n == math.floor(n) and n >= 0 then return n end
  return nil, "spin must be a whole number"
end)

local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

-- Split "a=1, b={x=2, y=3}" at top-level commas into {name, value} pairs;
-- braces around a value are removed. A key without "=" has value nil.
function M.parse_list(s)
  local out, depth, start = {}, 0, 1
  local function take(piece)
    piece = trim(piece)
    if piece == "" then return end
    local k, v = piece:match("^([^=]-)%s*=%s*(.*)$")
    if not k then
      out[#out + 1] = { trim(piece), nil }
      return
    end
    v = trim(v)
    local inner = v:match("^{(.*)}$")
    out[#out + 1] = { trim(k), inner and trim(inner) or v }
  end
  for i = 1, #s do
    local ch = s:sub(i, i)
    if ch == "{" then depth = depth + 1
    elseif ch == "}" then depth = depth - 1
      if depth < 0 then error("unbalanced braces in \"" .. s .. "\"") end
    elseif ch == "," and depth == 0 then
      take(s:sub(start, i - 1))
      start = i + 1
    end
  end
  if depth ~= 0 then error("unbalanced braces in \"" .. s .. "\"") end
  take(s:sub(start))
  return out
end

-- Parse a key=value list into a settings table. Errors name the key.
function M.parse(s)
  local t = {}
  for _, kv in ipairs(M.parse_list(s)) do
    local k, v = kv[1], kv[2]
    local parse = keys[k]
    if not parse then
      local known = {}
      for name in pairs(keys) do known[#known + 1] = name end
      table.sort(known)
      error(string.format("unknown key \"%s\" (known keys: %s)", k, table.concat(known, ", ")))
    end
    if v == nil then error(string.format("key \"%s\" needs a value", k)) end
    local val, msg = parse(v)
    if val == nil then error(string.format("%s: %s (got \"%s\")", k, msg, v)) end
    if k == "density" then
      t.mix = M.merge({ mix = t.mix }, { mix = { density = val } }).mix
    elseif k == "visitors" then
      local m = {}
      for _, name in ipairs(Mix.visitors) do m[name] = val end
      t.mix = M.merge({ mix = t.mix }, { mix = m }).mix
    else
      t[k] = M.merge({ [k] = t[k] }, { [k] = val })[k]
    end
  end
  return t
end

-- Merge settings table `over` onto `base`, returning a new table. Table
-- values (such as mix) merge key by key.
function M.merge(base, over)
  local t = {}
  for k, v in pairs(base or {}) do t[k] = v end
  for k, v in pairs(over or {}) do
    if type(v) == "table" and type(t[k]) == "table" then
      local m = {}
      for a, b in pairs(t[k]) do m[a] = b end
      for a, b in pairs(v) do m[a] = b end
      t[k] = m
    else
      t[k] = v
    end
  end
  return t
end

return M
