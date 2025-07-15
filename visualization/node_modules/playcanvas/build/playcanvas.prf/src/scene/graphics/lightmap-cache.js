import { RefCountedCache } from '../../core/ref-counted-cache.js';

class LightmapCache {
		static{
				this.cache = new RefCountedCache();
		}
		static incRef(texture) {
				this.cache.incRef(texture);
		}
		static decRef(texture) {
				this.cache.decRef(texture);
		}
		static destroy() {
				this.cache.destroy();
		}
}

export { LightmapCache };
