This file trains a network on the (x,y(x)) data where y(x)=sin(2x):

pairs = [
        [[0.01],[0.0099833]],
        [[0.05],[0.0998334]],
        [[0.1],[0.198669]],
        [[0.15],[0.29552]],
        [[0.2],[0.389418]],
        [[0.25],[0.479426]],
        [[0.3],[0.564642]],
        [[0.35],[0.644218]],
        [[0.4],[0.717356]],
        [[0.45],[0.783327]],
        [[0.5],[0.841471]],
        [[0.55],[0.891207]],
        [[0.6],[0.932039]],
        [[0.65],[0.963558]],
        [[0.7],[0.98545]],
        [[0.75],[0.997495]],
        [[0.8],[0.999574]],
        [[0.85],[0.991665]],
        [[0.9],[0.973848]],
        [[0.95],[0.9463]]
]

It then uses autograd to compute and plot the first and second derivatives of the network's output.

Curves are plotted to show y'(x)=2cos(2x) and y''(x)=-4sin(2x) for comparison.

The second derivative is hard to nail down, to a loss of 1e-5.
