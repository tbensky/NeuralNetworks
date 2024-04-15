import graphviz

def draw(NN):
    g = graphviz.Digraph("G", filename='nn.gv')
    g.attr(splines='false',rankdir="LR",fontsize="20pt",nodesep="0.5")
    g.attr('node', fontsize='12pt')

    # need to connect nodes in a layer with invisible line
    # to set proper tail->head ordering between nodes.

    #input layer
    with g.subgraph(name='cluster_0') as c:
        c.attr(peripheries = '0',fontsize="12pt")

        layer = 0
        for input_neuron in range(len(NN[layer])):
            z = NN[layer][input_neuron]['z']
            a = NN[layer][input_neuron]['a']
            delta = NN[layer][input_neuron]['delta']
            c.node(f"I{input_neuron}",label=f"{NN[layer][input_neuron]['desc']}\nz={z:.3f}\na={a:.3f}\ndelta={delta:.2e}")

        for i in range(len(NN[layer])-1):
            c.edge(f"I{i}",f"I{i+1}",style="invis")

        c.attr(label='Input layer')

    #hidden layers
    for layer in range(1,len(NN)-1):
        with g.subgraph(name=f"cluster_{layer}") as c:
            c.attr(peripheries = '0',fontsize="12pt")

            for neuron in range(0,len(NN[layer])):
                z = NN[layer][neuron]['z']
                a = NN[layer][neuron]['a']
                delta = NN[layer][neuron]['delta']
                c.node(f"H{layer}{neuron}",label=f"{NN[layer][neuron]['desc']}\nz={z:.3f}\na={a:.3f}\ndelta={delta:.2e}")

            for neuron in range(0,len(NN[layer])-1):
                c.edge(f"H{layer}{neuron}",f"H{layer}{neuron+1}",style="invis")

            c.attr(label=f"Hidden layer {layer}")

    #output layer
    layer = len(NN)-1
    with g.subgraph(name=f"cluster_{layer}") as c:
        c.attr(peripheries = '0',fontsize="12pt")

        for neuron in range(len(NN[layer])):
            z = NN[layer][neuron]['z']
            a = NN[layer][neuron]['a']
            delta = NN[layer][neuron]['delta']
            c.node(f"O{neuron}",label=f"{NN[layer][neuron]['desc']}\nz={z:.3f}\na={a:.3f}\ndelta={delta:.2e}")

        for neuron in range(len(NN[layer])-1):
            c.edge(f"O{neuron}",f"O{neuron+1}",style="invis")

        c.attr(label='Output layer')

    #now, make the connections

    #connect input and first hidden layer
    for input_neuron in range(len(NN[0])):
        for hidden_neuron in range(len(NN[1])):
            w = NN[0][input_neuron]['w'][hidden_neuron]
            dw = NN[0][input_neuron]['dw'][hidden_neuron]
            g.edge(f"I{input_neuron}",f"H{1}{hidden_neuron}",label=f"w={w:.3f}\ndw={dw:.2e}")

    
    #connect all hidden layers
    for layer in range(1,len(NN)-2):
        for neuron in range(len(NN[layer])):
            for neuron_next_layer in range(len(NN[layer+1])):
                w = NN[layer][neuron]['w'][neuron_next_layer]
                dw = NN[layer][neuron]['dw'][neuron_next_layer]
                g.edge(f"H{layer}{neuron}",f"H{layer+1}{neuron_next_layer}",label=f"w={w:.3f}\ndw={dw:.2e}")

    #connect last hidden layer with output layer
    last_layer = len(NN)-2
    for neuron in range(len(NN[last_layer])):
        for output_neuron in range(len(NN[last_layer+1])):
            w = NN[last_layer][neuron]['w'][output_neuron]
            dw = NN[last_layer][neuron]['dw'][output_neuron]
            g.edge(f"H{last_layer}{neuron}",f"O{output_neuron}",label=f"w={w:.3f}\ndw={dw:.2e}")

    
    g.view()
    f = open("net1.gv","w")
    f.write(g.source)
    f.close()

