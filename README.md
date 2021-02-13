# CopEnt.jl: Copula Entropy in Julia

This package is a Julia fork of the original author's R package ([copent](https://github.com/majianthu/copent)) and the Python package ([pycopent](https://github.com/majianthu/pycopent))

## WARNING

The function `copula_entropy` in the package *indeed* returns the copula entropy instead of the negative copula entropy, which is done by the corresponding `copent` function in copent or pycopent.

## Example

```julia
using Distributions, CopEnt
ρ = 0.6
μ = zeros(2)
Σ = [1 ρ; ρ 1]
x = rand(MvNormal(μ, Σ), 2_000)
mi = -copent(x) # this gives us 0.1849467947096306
                # true value is 0.2231435513142097
```

Also see the [example notebook](docs/example.ipynb) for some visualizations.
