# Neural Networks from scratch
On rappel la définition d'un matric jacobienne d'une fonction 
$$
\begin{align*}
    F   &: \mathbb{R}^n \longmapsto \mathbb{R}^m \\ 
        &:{\begin{pmatrix}x_{1}\\ \vdots \\ x_{n} \end{pmatrix}} \longmapsto {\begin{pmatrix} f_{1}(x_{1},\dots ,x_{n}) \\ \vdots \\ f_{m}(x_{1},\dots ,x_{n}) \end{pmatrix}}
\end{align*}
$$
Les dérivées partielles de ces fonctions en un point $M$, si elles existent, peuvent être rangées dans une matrice à $m$ lignes et $n$ colonnes, appelée matrice jacobienne de $F$ :
$$
J_{F}\left(M\right)={\begin{pmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{pmatrix}} \in \mathbb{R}^{n \times m}
$$

## Module linéaire
### Forward
Soit $z^{k-1} \in \mathbb{R}^{batch \times input}$, $ W \in \mathbb{R}^{input, ouput} $, $M_W (z^{k-1}) : \mathbb{R}^{batch \times input} \longmapsto \mathbb{R}^{batch \times output}$
$$
\begin{align*}
    M_W(z^{k-1})   &= z^{k-1} * W \\ 
                    &= \begin{pmatrix}
                        z^{k-1}_{1,1} & z^{k-1}_{1,2} & \cdots & z^{k-1}_{1, input} \\
                        z^{k-1}_{2,1} & z^{k-1}_{2,2} & \cdots & z^{k-1}_{2, input} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        z^{k-1}_{batch,1} & z^{k-1}_{batch,2} & \cdots & z^{k-1}_{batch, input} 
                    \end{pmatrix} * \begin{pmatrix}
                        w_{1,1} & w_{1,2} & \cdots & w_{1, output} \\
                        w_{2,1} & w_{2,2} & \cdots & w_{2, output} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        w_{input,1} & w_{input,2} & \cdots & w_{input, output} 
                    \end{pmatrix} \\
                    &= \begin{pmatrix}
                        \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, 1} & \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, 2} & \cdots & \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, output} \\
                        \sum_{i=1}^{input} z^{k-1}_{2,i} w_{i, 1} & \sum_{i=1}^{input} z^{k-1}_{2,i} w_{i, 2} & \cdots & \sum_{i=1}^{input} z^{k-1}_{2,i} w_{i, output} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        \sum_{i=1}^{input} z^{k-1}_{batch,i} w_{i, 1} & \sum_{i=1}^{input} z^{k-1}_{batch,i} w_{i, 2} & \cdots & \sum_{i=1}^{input} z^{k-1}_{batch,i} w_{i, output} \\
                    \end{pmatrix} \\
                    &= \begin{pmatrix}
                        z^{k}_{1,1} & z^{k}_{1,2} & \cdots & z^{k}_{1, output} \\
                        z^{k}_{2,1} & z^{k}_{2,2} & \cdots & z^{k}_{2, output} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        z^{k}_{batch,1} & z^{k}_{batch,2} & \cdots & z^{k}_{batch, output} 
                    \end{pmatrix} \\
                    &= z^{k} \in \mathbb{R}^{batch \times output} \\
\end{align*}
$$
Pour $batch=1$, on a 
$$
\begin{align*}
    M_W(z^{k-1})   &= z^{k-1} * W \\ 
                    &= \begin{pmatrix}
                        z^{k-1}_{1,1} & z^{k-1}_{1,2} & \cdots & z^{k-1}_{1, input}
                    \end{pmatrix} * \begin{pmatrix}
                        w_{1,1} & w_{1,2} & \cdots & w_{1, output} \\
                        w_{2,1} & w_{2,2} & \cdots & w_{2, output} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        w_{input,1} & w_{input,2} & \cdots & w_{input, output} 
                    \end{pmatrix} \\
                    &= \begin{pmatrix}
                        \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, 1} & \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, 2} & \cdots & \sum_{i=1}^{input} z^{k-1}_{1,i} w_{i, output} \\
                    \end{pmatrix} \\
                    &= \begin{pmatrix}
                        f_{1, w_{*, 1}}(z^{k-1}) & f_{2, w_{*, 2}}(z^{k-1}) & \cdots & f_{output, w_{*, output}}(z^{k-1})
                    \end{pmatrix} \\
                    &= \begin{pmatrix}
                        z^{k}_{1,1} & z^{k}_{1,2} & \cdots & z^{k}_{1, output} \\
                    \end{pmatrix} \\
                    &= z^{k} \in \mathbb{R}^{output} \\
\end{align*}
$$


### Dérivé du module par rapport aux paramètres
Prenons $batch=1$, ainsi $M_W (z^{k-1}) : \mathbb{R}^{input} \longmapsto \mathbb{R}^{output}$ et une écriture de $W$ tel que $W=(w_{*,1}, w_{*,2}, \cdots, w_{*,output})$ et $w_{*,i} \in \mathbb{R}^{input}$
$$
\begin{align*}
    J_{M_W}^{z^{k-1}} &= {\begin{pmatrix}
        {\dfrac {\partial f_{1, w_{*, 1}}}{\partial w_{*, 1}}} & \cdots & {\dfrac {\partial f_{1, w_{*, 1}}}{\partial w_{*, output}}} \\
        \vdots &\ddots &\vdots \\
        {\dfrac {\partial f_{output, w_{*, output}}}{\partial w_{*, 1}}} & \cdots & {\dfrac {\partial f_{output, w_{*, output}}}{\partial w_{*, output}}}
        \end{pmatrix}}
\end{align*}
$$
Mais ça c'est pour l'écriture réduite du $W$, il faut rester en écriture matriciel sinon c'est pas visualisable. 
$$
\begin{align*}
    J_{M_W}^{z^{k-1}}   &= 
        {\begin{pmatrix}
            {\dfrac {\partial (z^{k-1} * w_{*, 1})}{\partial w_{*, 1}}} & \cdots & {\dfrac {\partial (z^{k-1} * w_{*, 1})}{\partial w_{*, output}}} \\
            \vdots &\ddots &\vdots \\
            {\dfrac {\partial (z^{k-1} * w_{*, output})}{\partial w_{*, 1}}} & \cdots & {\dfrac {\partial (z^{k-1} * w_{*, output})}{\partial w_{*, output}}}
        \end{pmatrix}} \\
                        &= 
        \begin{pmatrix}
            z^{k-1} & 0         & \cdots & \cdots    & 0 \\
            0       & z^{k-1}   & 0      &           & 0 \\
            \vdots  & 0         & \ddots & \ddots    & \vdots  \\
            \vdots  &           & \ddots & \ddots    & 0  \\
            0       & \cdots    & \cdots & 0         & z^{k-1}    \\
        \end{pmatrix}
\end{align*}
$$
Or ici $f_{1, w_{*,1}}$ ne dépend pas de $w_{*, output}$, on a donc une matrice diagonale avec notre unique $z^{k-1}$ reproduit $output$ fois sur la diagonale .

Ca c'était pour un $batch=1$, pour un plus grands $batch$ j'imagine qu'il y a une 3ème dimension à la Jacobienne représentant le batch. 

### Dérivé du module par rapport aux entrées
Prenons $batch=1$, ainsi $M_W (z^{k-1}) : \mathbb{R}^{input} \longmapsto \mathbb{R}^{output}$ 
$$
\begin{align*}
    J_{M_W}^{z^{k-1}} &= {\begin{pmatrix}
        {\dfrac {\partial f_{1}}{\partial z^{k-1}_{1}}} & \cdots & {\dfrac {\partial f_{1}}{\partial z^{k-1}_{input}}} \\
        \vdots &\ddots &\vdots \\
        {\dfrac {\partial f_{output}}{\partial z^{k-1}_{1}}} & \cdots & {\dfrac {\partial f_{output}}{\partial z^{k-1}_{input}}}
        \end{pmatrix}} \\
                    &= \begin{pmatrix}
                        w_{1,1} & w_{2,1} & \cdots & w_{input,1} \\
                        w_{1,2} & w_{2,2} & \cdots & w_{input,2} \\
                        \vdots  & \vdots  & \ddots & \vdots  \\
                        w_{1,output} & w_{2,output} & \cdots & w_{input,output} \\
                    \end{pmatrix} \\ 
                    &= W
\end{align*}
$$
Ca c'était pour un $batch=1$, pour un plus grands $batch$ j'imagine qu'il y a une 3ème dimension à la Jacobienne représentant le batch. Dans ce cas elle est la réplication de W. 



## Idée : 
- un ReadTheDoc