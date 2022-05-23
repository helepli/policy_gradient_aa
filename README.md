# Policy Gradient REINFORCE with the Actor-Advisor (Policy Intersection + Learning correction)

The Actor-Advisor [[1]](#1) is a Policy Shaping method based on the Policy Intersection formula [[2]](#2), adapted to Policy Gradient methods. At acting time, the agent samples the mixture of its policy and of an advisory policy, following the Policy Shaping formula in [[2]](#2):

<img src="https://latex.codecogs.com/svg.latex?\Large&space;a_t\sim\pi_L(s_t)\times\pi_A(s_t)=\frac{\overbrace{\pi_L(s_t)\,\pi_A(s_t)}^{\text{element-wise product}}}{\underbrace{\pi_L(s_t)\cdot\pi_A(s_t)}_{\sum_{a\in A\pi_L(a|s_t)\pi_A(a|s_t)}}" title="\Large a_t\sim\pi_L(s_t)\times\pi_A(s_t)=\frac{\overbrace{\pi_L(s_t)\,\pi_A(s_t)}^{\text{element-wise product}}}{\underbrace{\pi_L(s_t)\cdot\pi_A(s_t)}_{\sum_{a\in A\pi_L(a|s_t)\pi_A(a|s_t)}}" />

The adapation required for Policy Gradient to allow an advisory policy to be mixed with the policy it is currently learning is to incoprorate the advisory policy in the loss:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t)\times\pi_A(a_t|s_t))" title="\Large loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t)\times\pi_A(a_t|s_t))" />

## References
<a id="1">[1]</a>
Helene plisnier
<a id="2">[2]</a>
Griffith et al Polocy Shaping

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
