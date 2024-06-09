import graphviz

g = graphviz.Digraph(comment='CNN')
g.attr('graph', rankdir='TB',splines='true')

#input data size (IxI)
I = 4
#kernal size (KxK)
K = 2
#stride or step in kernel sliding
stride = 1
#size of convolutional layer
R = int((I-K)/stride + 1)


#raw input
#build up Lenum to have a correspondence between Lnm values and a flattened L-vector
#Here L="layer"
Lcount = 0
Lenum = {}
for r in range(I):
    for c in range(I):
        Llabel = f"L{r}{c}"
        g.node(Llabel,f"L{r}{c}({Lcount})")
        Lenum[(r,c)] = Lcount
        Lcount += 1

#convolutional layer
#build up renum to have a correspondence between rnm and a flattened r-vector
rcount = 0
renum = {}
for r in range(R):
    for c in range(R):
        rlabel = f"r{r}{c}"
        g.node(rlabel,f"r{r}{c}({rcount})")
        renum[(r,c)] = rcount
        rcount += 1

cs = []
for r in range(K):
    for c in range(K):
        cs.append(f"c{r}{c}")

#Idea:
#
# Supposed kernel is array of cnm scalars:
# |---|---|
# |c00|c01|
# |---+---|
# |c10|c11|
# |---|---|
#
# Will be referred to as a weight number 0..3 like this
# |---|---|
# | 0 | 1 |
# |---+---|
# | 2 | 3 |
# |---|---|
#
#
#if kernel is flattened to c-vector: c=<c00,c01,c10,c11>
#and result of convolution is r as in
# |---|---|
# |r00|r01|
# |---+---|
# |r10|r11|
# |---|---|
#
#result of convolution elements rnm = c . d as in
#r00 = c . d00, r01 = c . d01, r10 = c . d01, r11=c . d11
#
# we need the d-vectors, which are: (row,col) indicies from I needed
# for the convolution. As a vector, it is a flattened version of the 2d matricies
# that are drawn from the input data, so we need to know what dnm's are 
# the (d-vectors) 
#
#I = input data size
#K = kernel size
#
#row and col are where to place the top left edge of
#the KxK kernel in the input data IxI
Lcount = 0
rcount = 0


weight_dict = {}
for row in range(I):
    for col in range(I):
        weight_dict[(row,col)] = {"neuron": Lenum[(row,col)],"forward_neurons": [],"lut": {}}


#now, loop through components of the r-vector
#or each r-neuron
rvec = 0
for row in range(0,I-K+1,stride):
    for col in range(0,I-K+1,stride):
        d = []
        #go through the kernel and generate (row,col)
        #coordinates in the input data grid. Compile them as
        #components of the d-vector
        for krow in range(K):
            for kcol in range(K):
                d.append((row + krow,col + kcol))
        #each component of the d-vector tell us what input neuron (n,m)
        #is associated with computing this component of the r-vector 
        for (i,dcomp) in enumerate(d):
            weight_dict[dcomp]['forward_neurons'].append({"neuron_number": rvec,"weight_number": i})
            #look up table (lut), mapping back later neuron (dcomp) to the forward layer neuron # (rvec) 
            # via the convolutional weight (i) 
            weight_dict[dcomp]['lut'][rvec] = i
        rvec += 1

        for i,ij in enumerate(d):
            rlab = f"r{row}{col}"
            Llab = f"L{ij[0]}{ij[1]}"
            g.edge(Llab,rlab,f"{cs[i]}")
         

for w in weight_dict:
    print(w,weight_dict[w])

g.view()
f = open("cnn.gv","w")
f.write(g.source)
f.close()
