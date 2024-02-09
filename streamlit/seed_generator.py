#!/usr/bin/env python
# coding: utf-8

# In[1]:


def custom_hash(seed_str):
    hash_value = 0
    for char in seed_str:
        hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFF
    return hash_value

def generate_seeds(base_seed, num_seeds):
    seeds = []
    for i in range(num_seeds):
        seed_str = str(base_seed) + str(i)
        seed_hash = custom_hash(seed_str)
        seeds.append(seed_hash)
    return seeds

