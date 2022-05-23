# Policy Gradient REINFORCE with the Actor-Advisor (Policy Intersection + Learning correction)

The Actor-Advisor [[1]](#1) is a Policy Shaping method based on the Policy Intersection formula [[2]](#2), adapted to Policy Gradient methods. This adapation consists in modifying the loss of Policy Gradient to insorporate the policy of the advisor (pi_A) when updating the actor's policy (pi_L): loss = - sum(G_t log (pi_L(a_t, s_t) times p_A(a_t, s_t)))


<img src="https://latex.codecogs.com/svg.latex?\Large&space;loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t))" title="\Large loss=-\sum\limits_{t=0}^{T}G_t\log(\pi_{\theta}(a_t|s_t))" />

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
