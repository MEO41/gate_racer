"""Train a gate-racing policy with TF-Agents PPO (settings from Swift, Methods).

    python train_ppo.py                       # 2400 iters x 100 drones x 100 steps = 24M steps
"""
import argparse, os, time
import numpy as np
import tensorflow as tf
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.utils import common

from tf_agents.environments import tf_py_environment

from drone_race.tf_env import make_tf_env, DroneRaceEnv


def leaky(x):
    return tf.nn.leaky_relu(x, alpha=0.2)


def policy_part(x):
    return x[..., :31]


def build_agent(env, lr=3e-4, total_updates=None, privileged=False):
    """privileged: observation = [policy obs (31), true-state obs (31)]; the actor only sees the
    first half, the critic sees both (asymmetric actor-critic, as in Swift)."""
    obs_spec, act_spec = env.observation_spec(), env.action_spec()
    actor = actor_distribution_network.ActorDistributionNetwork(
        obs_spec, act_spec, fc_layer_params=(128, 128), activation_fn=leaky,
        preprocessing_layers=tf.keras.layers.Lambda(policy_part) if privileged else None)
    critic = value_network.ValueNetwork(
        obs_spec, fc_layer_params=(128, 128), activation_fn=leaky)
    agent = ppo_clip_agent.PPOClipAgent(
        env.time_step_spec(), act_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=(
            tf.keras.optimizers.schedules.PolynomialDecay(lr, total_updates, lr * 0.1)
            if total_updates else lr)),
        actor_net=actor, value_net=critic,
        importance_ratio_clipping=0.2, discount_factor=0.99,
        lambda_value=0.95, entropy_regularization=0.0,
        num_epochs=10, use_gae=True, use_td_lambda_return=True,
        normalize_observations=True, normalize_rewards=False,
        gradient_clipping=0.5, train_step_counter=tf.Variable(0))
    agent.initialize()
    return agent


def evaluate(policy, env, pyenv, steps=1000):
    """Greedy policy, fixed seed. Score = mean gates passed per drone before its first crash."""
    pyenv.sim.rng = np.random.default_rng(7)
    if pyenv.sim.perception is not None:
        pyenv.sim.perception.rng = np.random.default_rng(8)
    t = env.reset()
    alive = np.ones(pyenv.n, bool)
    gates = np.zeros(pyenv.n)
    act = common.function(policy.action)
    for _ in range(steps):
        t = env.step(act(t).action)
        info = pyenv.info
        gates += info['passed'] & alive
        alive &= ~info['crashed']
    return gates.mean(), alive.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num_envs', type=int, default=100)
    ap.add_argument('--steps_per_iter', type=int, default=100, help='steps per env per iteration')
    ap.add_argument('--iterations', type=int, default=2400)
    ap.add_argument('--eval_every', type=int, default=50)
    ap.add_argument('--perception', action='store_true', help='policy sees VIO + detector estimate, not the truth')
    ap.add_argument('--finetune_from', default=None, help='checkpoint dir of a trained agent')
    ap.add_argument('--privileged', action='store_true', help='critic also sees the true state')
    ap.add_argument('--init_policy', default=None, help='saved policy dir to copy weights from')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', default='runs/ppo')
    args = ap.parse_args()

    env = make_tf_env(args.num_envs, perception=args.perception, privileged=args.privileged)
    pyenv = env.pyenv
    agent = build_agent(env, lr=args.lr, total_updates=args.iterations * 10, privileged=args.privileged)
    if args.init_policy:
        # start from a saved policy (e.g. the best evaluated snapshot): actor, critic, obs normaliser
        import tensorflow_probability  # noqa: F401, registers distribution specs for loading
        src = {v.name: v for v in tf.saved_model.load(args.init_policy).model_variables}
        dst = {v.name: v for v in list(agent.policy.variables()) + list(agent._value_net.variables)}
        missing = [n for n in src if n not in dst]
        assert not missing, missing
        for n, v in src.items():
            d = dst[n]
            if d.shape == v.shape:
                d.assign(v)
            elif len(v.shape) == 1:          # observation normaliser stats: same stats for both halves
                d.assign(tf.concat([v, v], 0))
            else:                            # critic input layer: true-state half starts at zero
                d.assign(tf.concat([v, tf.zeros_like(v)], 0))
        print(f'initialised {len(src)} variables from', args.init_policy)
    if args.finetune_from:
        # start from a trained agent's weights (incl. observation normaliser), fresh step counters
        common.Checkpointer(ckpt_dir=args.finetune_from, agent=agent).initialize_or_restore().expect_partial()
        agent.train_step_counter.assign(0)
        agent._optimizer.iterations.assign(0)
        print('fine-tuning from', args.finetune_from)
    eval_py = DroneRaceEnv(num_envs=32, max_steps=10**9, perception=args.perception, privileged=args.privileged)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py)
    best = -1.0

    buf = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec, batch_size=args.num_envs, max_length=args.steps_per_iter + 1)
    driver = dynamic_step_driver.DynamicStepDriver(
        env, agent.collect_policy, observers=[buf.add_batch],
        num_steps=args.num_envs * args.steps_per_iter)
    driver.run = common.function(driver.run, autograph=False)
    train = common.function(agent.train, autograph=False)

    ckpt = common.Checkpointer(ckpt_dir=os.path.join(args.out, 'ckpt'), max_to_keep=3,
                               agent=agent, policy=agent.policy)
    ckpt.initialize_or_restore()
    saver = policy_saver.PolicySaver(agent.policy)

    ts_ = env.reset(); ps = agent.collect_policy.get_initial_state(args.num_envs)
    total, t0 = 0, time.time()
    for it in range(1, args.iterations + 1):
        ts_, ps = driver.run(ts_, ps)
        exp = buf.gather_all()
        loss = train(exp).loss
        buf.clear()
        total += args.num_envs * args.steps_per_iter
        r = exp.reward.numpy()
        if it % 5 == 0 or it == 1:
            gates, crash, neps = pyenv.pop_stats()
            print(f'it {it:4d} | steps {total/1e6:6.2f}M | r/step {r.mean():+.4f} | '
                  f'episodes {neps:4d} | gates/ep {gates:6.2f} | crash frac {crash:.2f} | '
                  f'loss {float(loss):7.3f} | {total/(time.time()-t0):,.0f} sps', flush=True)
        if it % 25 == 0 or it == args.iterations:
            ckpt.save(agent.train_step_counter)
            saver.save(os.path.join(args.out, 'policy'))
        if it % args.eval_every == 0 or it == args.iterations:
            score, survive = evaluate(agent.policy, eval_env, eval_py)
            tag = ''
            if score > best:
                best = score
                saver.save(os.path.join(args.out, 'policy_best'))
                tag = '  <- new best, saved policy_best'
            print(f'  eval: gates in 10 s {score:5.2f} | survived {survive:.2f}{tag}', flush=True)
    print('done', time.time() - t0, 's')


if __name__ == '__main__':
    main()
