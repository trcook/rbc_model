
import argparse
import json
import logging
import os
import time


import gym


from tensorforce.contrib.openai_gym import OpenAIGym


from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import datetime,os,sys

from gym.envs.registration import register



parser=argparse.ArgumentParser()
parser.add_argument("--subdir",default="./template",help="this subdir",dest='subdir')
parser.add_argument("--episodes",default=15000,help="num episodes",dest='episodes')
parser.add_argument("--max_timesteps",default=20,help="this subdir",dest='timesteps')
parser.add_argument("--agent_file",default="agent_config.json",help="this subdir",dest="agent_file")
parser.add_argument("--actor_file",default="actor_net.json",help="this subdir",dest="actor_file")
parser.add_argument("--env",default="RBCGym-v0",help="environment",dest="env")
parser.add_argument("--logdir",default="Graph",help="logdir",dest="logdir")

logging.basicConfig(format='|%(levelname)s| %(asctime)-15s (%(name)s) %(lineno)d:> %(message)s',datefmt='%m/%d/%Y %H:%M:%S')

logger=logging.getLogger("build_grid")
logger.setLevel(logging.DEBUG)
ld=logger.debug
lw=logger.warning
li=logger.info




def get_files(subdir,agent_file="trpo_config.json",actor_file="actor_net.json",logdir='Graph'):
    get_file=lambda x:os.path.abspath(os.path.join(subdir, x))
    agent_file=get_file(agent_file)
    li(agent_file)
    actor_file=get_file(actor_file)
    li(actor_file)


    load_file=lambda x:json.load(open(x,'r'))
    actor=load_file(actor_file)

    agent_spec=load_file(agent_file)

    nowish = datetime.date.strftime(datetime.datetime.now(), "%m%d%H%M%S")
    agent_spec['summarizer']['directory']='./Graph/{}'.format(subdir)
    agent_spec['saver']['directory']='./Graph/{}'.format(subdir)
    return actor,agent_spec

def build(agent_spec,actor,env):
    agent = Agent.from_spec(
    spec=agent_spec,
    kwargs=dict(
        states=env.states,
        actions=env.actions,
        network=actor
    )
    )
    runner = Runner(
        agent=agent,
        environment=env,
        repeat_actions=1
    )
    return runner,agent






def episode_finished(r,report_episodes=100):
    if r.episode % report_episodes == 0:
        steps_per_second = r.timestep / (time.time() - r.start_time)
        logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
            r.agent.episode, r.episode_timestep, steps_per_second
        ))

        logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
        logger.info("Average of last 500 rewards: {:0.2f}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
        logger.info("Average of last 100 rewards: {:0.2f}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
    return True



if __name__=='__main__':
    #! /usr/bin/env python3

    args = parser.parse_args()
    register(id=args.env,entry_point='mygym:MyGym')

    env=OpenAIGym(args.env)
    a,ag=get_files(subdir=args.subdir,actor_file=args.actor_file,agent_file=args.agent_file)
    runner,agent=build(ag,a,env)

    logger=logger
    logger.info("-" * 16)
    logger.info("Configuration:")
    logger.info(agent)
    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    ep_done=lambda x:episode_finished(x,report_episodes=100)
    runner.run(num_episodes=int(args.episodes),max_episode_timesteps=int(args.timesteps)-1,episode_finished=ep_done,deterministic=True)

    agent.save_model()
    env=gym.make(args.env)
    s=env.reset()
    out=[]
    for i in range(args.timesteps):
        a=agent.act(s,deterministic=True)
        e=env.step(a)
        out.append(env.render(mode='human'))
        s=e[0]
    with open("{}/output.txt".format(args.subdir),'w') as f:
        for i in out:
            f.write("{}\n".format(i))


    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


