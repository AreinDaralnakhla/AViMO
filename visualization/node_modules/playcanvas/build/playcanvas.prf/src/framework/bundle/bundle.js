import { EventHandler } from '../../core/event-handler.js';

class Bundle extends EventHandler {
		static{
				this.EVENT_ADD = 'add';
		}
		static{
				this.EVENT_LOAD = 'load';
		}
		addFile(url, data) {
				if (this._index.has(url)) {
						return;
				}
				this._index.set(url, data);
				this.fire('add', url, data);
		}
		has(url) {
				return this._index.has(url);
		}
		get(url) {
				return this._index.get(url) || null;
		}
		destroy() {
				this._index.clear();
		}
		set loaded(value) {
				if (!value || this._loaded) {
						return;
				}
				this._loaded = true;
				this.fire('load');
		}
		get loaded() {
				return this._loaded;
		}
		constructor(...args){
				super(...args), this._index = new Map(), this._loaded = false;
		}
}

export { Bundle };
