class Tracing {
		static{
				this._traceChannels = new Set();
		}
		static{
				this.stack = false;
		}
		static set(channel, enabled = true) {}
		static get(channel) {
				return Tracing._traceChannels.has(channel);
		}
}

export { Tracing };
