import { LAYERID_WORLD } from '../constants.js';

class BatchGroup {
		constructor(id, name, dynamic, maxAabbSize, layers = [
				LAYERID_WORLD
		]){
				this._ui = false;
				this._sprite = false;
				this._obj = {
						model: [],
						element: [],
						sprite: [],
						render: []
				};
				this.id = id;
				this.name = name;
				this.dynamic = dynamic;
				this.maxAabbSize = maxAabbSize;
				this.layers = layers;
		}
		static{
				this.MODEL = 'model';
		}
		static{
				this.ELEMENT = 'element';
		}
		static{
				this.SPRITE = 'sprite';
		}
		static{
				this.RENDER = 'render';
		}
}

export { BatchGroup };
