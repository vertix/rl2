from Geometry import BuildPath
from heapq import heappush
from heapq import heappop

class Cache(object):
    instance = None
    def __init__(self, ttl=10):
        self.paths = {}
        self.times_and_keys = []
        self.ttl = ttl
        
        
    @classmethod
    def GetInstance(cls)
        if instance is None:
            instance = Cache()
        return Cache.instance
    
    def InvalidateOldCache(self, tick):
        while self.times_and_keys and (self.times_and_keys[0][0] < tick - self.ttl):
            _, key = heappop(self.times_and_keys)
            self.paths.pop(key, None)
            
    def AddEntry(self, key, value, tick):
        self.paths[key] = value
        heappush(self.times_and_keys, (tick, key))
    
    def GetEntryFor(self, key, me, u, game, world):
        self.InvalidateOldCache(world.tick_index)
        if key in paths:
            return paths[key]
        path = BuildPath(me, target, game, world)
        self.AddEntry(key, path, world.tick_index)
        return path        
        
    def BuildKey(self, u):
        if hasattr(u, id) and u.id:
            return str(u.id)
        return 'x:%d,y:%d,r:%d' % (((int(u.x) + 4) / 5) * 5,
                                   ((int(u.y) + 4) / 5) * 5, 
                                   int(u.radius))
        
    def GetPathToTarget(self, me, t, game, world):
        key = BuildKey(t)
        return self.GetEntryFor(key, me, t, game, world)
    