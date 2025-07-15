import { EventHandler } from '../../core/event-handler.js';
import { SCRIPT_INITIALIZE, SCRIPT_POST_INITIALIZE } from './constants.js';

class Script extends EventHandler {
		static{
				this.EVENT_ENABLE = 'enable';
		}
		static{
				this.EVENT_DISABLE = 'disable';
		}
		static{
				this.EVENT_STATE = 'state';
		}
		static{
				this.EVENT_DESTROY = 'destroy';
		}
		static{
				this.EVENT_ATTR = 'attr';
		}
		static{
				this.EVENT_ERROR = 'error';
		}
		constructor(args){
				super();
				this.initScript(args);
		}
		set enabled(value) {
				this._enabled = !!value;
				if (this.enabled === this._enabledOld) return;
				this._enabledOld = this.enabled;
				this.fire(this.enabled ? 'enable' : 'disable');
				this.fire('state', this.enabled);
				if (!this._initialized && this.enabled) {
						this._initialized = true;
						this.fire('preInitialize');
						if (this.initialize) {
								this.entity.script._scriptMethod(this, SCRIPT_INITIALIZE);
						}
				}
				if (this._initialized && !this._postInitialized && this.enabled && !this.entity.script._beingEnabled) {
						this._postInitialized = true;
						if (this.postInitialize) {
								this.entity.script._scriptMethod(this, SCRIPT_POST_INITIALIZE);
						}
				}
		}
		get enabled() {
				return this._enabled && !this._destroyed && this.entity.script.enabled && this.entity.enabled;
		}
		initScript(args) {
				const script = this.constructor;
				this.app = args.app;
				this.entity = args.entity;
				this._enabled = typeof args.enabled === 'boolean' ? args.enabled : true;
				this._enabledOld = this.enabled;
				this.__destroyed = false;
				this.__scriptType = script;
				this.__executionOrder = -1;
		}
		static{
				this.__name = null;
		}
		static{
				this.__getScriptName = getScriptName;
		}
		static get scriptName() {
				return this.__name;
		}
}
const funcNameRegex = /^\s*function(?:\s|\s*\/\*.*\*\/\s*)+([^(\s\/]*)\s*/;
function getScriptName(constructorFn) {
		if (typeof constructorFn !== 'function') return undefined;
		if (constructorFn.scriptName) return constructorFn.scriptName;
		if ('name' in Function.prototype) return constructorFn.name;
		if (constructorFn === Function || constructorFn === Function.prototype.constructor) return 'Function';
		const match = `${constructorFn}`.match(funcNameRegex);
		return match ? match[1] : undefined;
}

export { Script, getScriptName };
