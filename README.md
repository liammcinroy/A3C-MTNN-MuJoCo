# A3C-MTNN-MuJoCo
Continuous [Asynchronous Advantage Actor Critic](https://arxiv.org/abs/1602.01783) implementation using [MuJoCo](http://www.mujoco.org/) and [MTNN](https://github.com/liammcinroy/MetaTemplateNeuralNet)

Parallel will need a few slight modifications, but just to the `*SimEnvi` classes (add a separate `mjData` instance for each instance of the `*SimEnvi`, which needs an instance for each thread. If I get MuJoCo Pro then I will update)

Also has code for human demonstration training (but more specifically [Human Checkpoint Replay](https://arxiv.org/abs/1607.05077))
