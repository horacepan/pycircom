import os
from collections import defaultdict
from gen_fc import gen_fc

CIRCUITDIR = './'
REL_CIRCUITDIR = '../circuits'
GENERATED_CIRCUIT_DIR = '../circuits/generated/'
MODULE_PATH_DIR = {
    'fc': os.path.join(CIRCUITDIR, 'fc.circom'),
    'relu': os.path.join(CIRCUITDIR, 'relu.circom'),
    'relu2d': os.path.join(CIRCUITDIR, 'relu.circom'),
    'square2d': os.path.join(CIRCUITDIR, 'square.circom'),
    'flatten': os.path.join(CIRCUITDIR, 'utils.circom')
}
COMPONENT_IDS = defaultdict(lambda: 1) # gives the next avail id for a circuit

def create_circom_file(circom_str, fname):
    with open(fname, 'w') as f:
        f.write(circom_str)

def gen_imports(spec):
    lines = []
    paths = set()

    for l in spec['layers']:
        paths.add(MODULE_PATH_DIR[l['type']])

    for path in paths:
        l = f'include "{path}";'
        lines.append(l)
    lines.append("\n");
    return lines

def gen_template_open(spec):
    line = f"template {spec['name']} ({spec['circ_args']}) {{"
    return line

def gen_template_close(spec):
    return "}"

def pipe_prev2curr_2d(prev, curr, rows, cols):
    fmt = [
        f"for (var i=0; i<{rows}; i++) {{",
        f"   for (var j=0; j<{cols}; j++) {{",
        f"      {curr}[i][j] <== {prev}[i][j];",
        f"   }}",
        f"}}",
    ]
    return ''.join(fmt)

def gen_comp_name(name):
    '''
    Fixed standardized formatting
    '''
    return f'_{type}'

def assign_layer_name(layer, model_name):
    lid = COMPONENT_IDS[layer['type']]
    COMPONENT_IDS[layer['type']] += 1
    return f"{model_name}_{layer['type']}{lid}"

def gen_layers(spec):
    '''
    '''
    deps = []
    lines = [
        f'    signal input in[n][{spec["in_dim"]}];',
        f'    signal output out[n][{spec["out_dim"]}];',
        ''
    ]
    prev_comp_out = 'in'
    prev_out_dim = spec['in_dim']
    prev_pipe = "in"
    for idx, layer in enumerate(spec['layers']):
        if layer['type'] == 'fc':
            layer_name = assign_layer_name(layer, spec['name']) # the
            circuit_name = layer_name
            circom_fname = os.path.join(GENERATED_CIRCUIT_DIR, f"{circuit_name}.circom") # path of file
            comp_name = f'_{layer_name}' # name of the component. ex: component _fc1 = model_fc1();
            layer_str = gen_fc(layer['weight'], layer['bias'], layer_name)
            #create_circom_file(layer_str, circom_fname) # should probably be saved elsewhere
            # put in the stuff that actually needs to occur in this model's template file
            deps.append(layer_str + "\n")
            lines.append(f"    component {comp_name} = {circuit_name}(n);")
            lines.extend([
                 "    for (var i = 0; i < n; i++) {",
                f"        for (var j = 0; j < {layer['weight'].shape[1]}; j++) {{",
                f"            {comp_name}.in[i][j] <== {prev_pipe}[i][j];",
                 "        }",
                 "    }",
                 ""
            ])
            prev_out_dim = layer['weight'].shape[0]
        elif layer['type'] == 'relu2d' or layer['type'] == 'square2d':
            layer_name = assign_layer_name(layer, spec['name']) # the
            circuit_name = layer['type']
            comp_name = f'_{layer_name}'
            # need the previous layer's output size
            lines.append(f"    component {comp_name} = {circuit_name}(n, {prev_out_dim});")
            lines.extend([
                 "    for (var i = 0; i < n; i++) {",
                f"        for (var j = 0; j < {prev_out_dim}; j++) {{",
                f"            {comp_name}.in[i][j] <== {prev_pipe}[i][j];",
                 "        }",
                 "    }",
                 ""
            ])
        prev_pipe = f"{comp_name}.out"
        prev_comp = comp_name
        prev_comp_out = "{comp_name}.out"
    # fill the output
    lines.extend([
             "    for (var i = 0; i < n; i++) {",
            f"        for (var j = 0; j < {prev_out_dim}; j++) {{",
            f"            out[i][j] <== {prev_pipe}[i][j];",
             "        }",
             "    }",
             ""
    ])

    return lines, deps

def gen_net(spec):
    '''
    feed forward circuit (previous layer directly -> proceeding layer)
    can be split into:
    - preamble (pragma shit)
    - imports
    - template definition
    - template body
    - template close ("}}")
    '''
    header = f'pragma circom {spec["version"]};\n'
    template_open = gen_template_open(spec)
    imports = gen_imports(spec) # needs to come after layers bc we may have to autogen some circom code!
    layers, circ_deps = gen_layers(spec)
    template_close = gen_template_close(spec)

    lines = [header] + imports + circ_deps + [template_open] + layers + [template_close]
    circuit_str = ('\n'.join(lines))
    fname = os.path.join(REL_CIRCUITDIR, f"{spec['name']}.circom")
    create_circom_file(circuit_str, fname)
    print('Generated circuit:', fname)
