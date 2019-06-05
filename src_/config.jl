# config parser

"""
    read_whaleconf(cfile::String)

Read the Whale configuration to a dictionary.
"""
function read_whaleconf(cfile::String)
    conf = defaultconf()
    if !(isfile(cfile))
        @warn "Not a file ``$cfile`; will use the default configuration"
        return conf
    end
    s = open(cfile, "r") do f
        curr_section = ""
        for line in eachline(f)
            line = strip(split(line, "#")[1])
            line = replacements(line)
            if isempty(line) || startswith(line,'#') || startswith(line,';')
                continue
            end
            if startswith(line, "[")
                curr_section = split(split(line, "[")[2], "]")[1]
                if !haskey(conf, curr_section)
                    conf[curr_section]= Dict()
                end
                continue
            end
            if curr_section == ""
                continue
            end
            if !occursin("=", line)
                continue
            end
            par_val = [strip(x) for x in split(line, "=")]
            par = par_val[1]
            val = [trytoparse(Float64, x) for x in split(par_val[2])]
            conf[curr_section][par] = length(val) > 1 ? Tuple(val) : val[1]
        end
    end
    return conf
end

# default configuration
defaultconf() = Dict{String,Dict{String,Any}}(
    "slices" => Dict("length"=>0.05, "min"=>5, "max"=>Inf),
    "wgd"    => Dict{String,Tuple}()
)

# replace greek characters
function replacements(line::AbstractString)
    line = replace(line, "λ" => "l")
    line = replace(line, "μ" => "m")
    line = replace(line, "η" => "e")
    line = replace(line, "θ" => "r")
    line = replace(line, "ν" => "v")
    return line
end

# try to parse a string to some type, else return just the string
trytoparse(T, x) = tryparse(T, x) == nothing ? string(x) : tryparse(T, x)
