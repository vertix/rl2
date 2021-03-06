from Geometry import BuildPath
from heapq import heappush
from heapq import heappop

CACHE_TTL=30

class Cache(object):
    instance = None
    def __init__(self, ttl=CACHE_TTL):
        self.paths = {}
        self.times_and_keys = []
        self.ttl = ttl

    @classmethod
    def GetInstance(cls):
        if Cache.instance is None:
            Cache.instance = Cache()
        return Cache.instance

    def InvalidateOldCache(self, tick):
        while self.times_and_keys and (self.times_and_keys[0][0] < tick - self.ttl):
            _, key = heappop(self.times_and_keys)
            self.paths.pop(key, None)

    def AddEntry(self, key, value, tick):
        self.paths[key] = value
        heappush(self.times_and_keys, (tick, key))

    def GetEntryFor(self, key, me, t, state):
        self.InvalidateOldCache(state.world.tick_index)
        if key in self.paths:
            return self.paths[key]
        path = BuildPath(me, t, state)
        self.AddEntry(key, path, state.world.tick_index)
        return path        
        
    def BuildKey(self, u):
        if hasattr(u, 'id') and u.id:
            return str(u.id)
        key = 'x:%d,y:%d' % (((int(u.x) + 4) / 5) * 5,
                                   ((int(u.y) + 4) / 5) * 5)
        if hasattr(u, 'radius'):
            key += 'r:%d' % int(u.radius)
        return key
        
    def GetPathToTarget(self, me, t, state):
        key = self.BuildKey(t)
        return self.GetEntryFor(key, me, t, state)
    