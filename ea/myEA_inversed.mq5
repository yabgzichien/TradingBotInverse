#property strict

input string InpSymbols = "BTCUSD,XAUUSD";
input ENUM_TIMEFRAMES InpAnchorTF = PERIOD_H4;
input ENUM_TIMEFRAMES InpExecutionTF = PERIOD_M15;
input int InpExportBars = 1500;
input int InpTimerSec = 15;

string _symbols[];

string _rates_file(string symbol, ENUM_TIMEFRAMES tf)
{
   string tf_s = EnumToString(tf);
   StringReplace(tf_s, "PERIOD_", "");
   return "ag_rates_" + symbol + "_" + tf_s + ".csv";
}

string _commands_file()
{
   return "ag_commands.csv";
}

bool _split_symbols(string s, string &out[])
{
   StringReplace(s, ";", ",");
   int n = StringSplit(s, ',', out);
   if(n <= 0) return false;
   for(int i=0;i<n;i++)
   {
      StringTrimLeft(out[i]);
      StringTrimRight(out[i]);
   }
   return true;
}

bool _write_rates_csv(string symbol, ENUM_TIMEFRAMES tf, int bars)
{
   MqlRates rates[];
   int copied = CopyRates(symbol, tf, 0, bars, rates);
   if(copied <= 0) return false;

   int handle = FileOpen(_rates_file(symbol, tf), FILE_WRITE|FILE_CSV|FILE_COMMON);
   if(handle == INVALID_HANDLE) return false;

   FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume");

   for(int i=copied-1; i>=0; i--)
   {
      FileWrite(
         handle,
         (long)rates[i].time,
         DoubleToString(rates[i].open, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].high, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].low, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         DoubleToString(rates[i].close, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
         (long)rates[i].tick_volume,
         (long)rates[i].spread,
         (long)rates[i].real_volume
      );
   }

   FileClose(handle);
   return true;
}

double _calc_lots_by_risk(string symbol, double entry, double sl, double risk_usd)
{
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double vol_min = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vol_max = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(tick_size <= 0.0) tick_size = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(tick_size <= 0.0) tick_size = 0.01;
   if(tick_value <= 0.0) return vol_min;

   double sl_dist = MathAbs(entry - sl);
   double sl_ticks = sl_dist / tick_size;
   if(sl_ticks <= 0.0) return vol_min;

   double lots = risk_usd / (sl_ticks * tick_value);
   if(vol_step > 0.0)
      lots = MathFloor(lots / vol_step) * vol_step;

   if(lots < vol_min) lots = vol_min;
   if(lots > vol_max) lots = vol_max;
   return lots;
}

bool _cancel_all_pending(string symbol, long magic)
{
   int total = OrdersTotal();
   for(int i=total-1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if(OrderGetString(ORDER_SYMBOL) != symbol) continue;
      if((long)OrderGetInteger(ORDER_MAGIC) != magic) continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type != ORDER_TYPE_BUY_LIMIT && type != ORDER_TYPE_SELL_LIMIT && type != ORDER_TYPE_BUY_STOP && type != ORDER_TYPE_SELL_STOP) continue;

      MqlTradeRequest req;
      MqlTradeResult res;
      ZeroMemory(req);
      ZeroMemory(res);
      req.action = TRADE_ACTION_REMOVE;
      req.order = ticket;
      req.symbol = symbol;
      OrderSend(req, res);
   }
   return true;
}

bool _has_equivalent_pending(string symbol, long magic, int desired_type, double entry, double sl, double tp, double tol)
{
   int total = OrdersTotal();
   for(int i=0; i<total; i++)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if(OrderGetString(ORDER_SYMBOL) != symbol) continue;
      if((long)OrderGetInteger(ORDER_MAGIC) != magic) continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if((int)type != desired_type) continue;
      if(MathAbs(OrderGetDouble(ORDER_PRICE_OPEN) - entry) > tol) continue;
      if(MathAbs(OrderGetDouble(ORDER_SL) - sl) > tol) continue;
      if(MathAbs(OrderGetDouble(ORDER_TP) - tp) > tol) continue;
      return true;
   }
   return false;
}

bool _place_limit(string symbol, long magic, int dir, double entry, double sl, double tp, double risk_usd, double replace_tol_points)
{
   if(!SymbolSelect(symbol, true)) return false;
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double tol = replace_tol_points * point;

   double current_price = (dir == 1) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
   int desired_type;
   if(dir == 1) {
      if(entry > current_price) desired_type = ORDER_TYPE_BUY_STOP;
      else desired_type = ORDER_TYPE_BUY_LIMIT;
   } else {
      if(entry < current_price) desired_type = ORDER_TYPE_SELL_STOP;
      else desired_type = ORDER_TYPE_SELL_LIMIT;
   }

   if(_has_equivalent_pending(symbol, magic, desired_type, entry, sl, tp, tol))
      return true;

   _cancel_all_pending(symbol, magic);

   double lots = _calc_lots_by_risk(symbol, entry, sl, risk_usd);

   double stops_level_points = (double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double min_stop_dist = stops_level_points * point;

   if(dir == 1)
   {
      if(!(sl < entry && entry < tp)) {
         PrintFormat("Validation failed (BUY): sl(%.5f) < entry(%.5f) < tp(%.5f) is FALSE", sl, entry, tp);
         return false;
      }
      if(min_stop_dist > 0.0)
      {
         if((entry - sl) < min_stop_dist) sl = entry - min_stop_dist;
         if((tp - entry) < min_stop_dist) tp = entry + min_stop_dist;
      }
   }
   else
   {
      if(!(tp < entry && entry < sl)) {
         PrintFormat("Validation failed (SELL): tp(%.5f) < entry(%.5f) < sl(%.5f) is FALSE", tp, entry, sl);
         return false;
      }
      if(min_stop_dist > 0.0)
      {
         if((sl - entry) < min_stop_dist) sl = entry + min_stop_dist;
         if((entry - tp) < min_stop_dist) tp = entry - min_stop_dist;
      }
   }

   entry = NormalizeDouble(entry, digits);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   MqlTradeRequest req;
   MqlTradeResult res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action = TRADE_ACTION_PENDING;
   req.symbol = symbol;
   req.volume = lots;
   req.type = (ENUM_ORDER_TYPE)desired_type;
   req.price = entry;
   req.sl = sl;
   req.tp = tp;
   req.magic = magic;
   req.type_time = ORDER_TIME_GTC;
   req.type_filling = ORDER_FILLING_RETURN;

   bool ok = OrderSend(req, res);
   if (!ok) {
       PrintFormat("OrderSend failed for %s. Retcode: %d. Error: %d", symbol, res.retcode, GetLastError());
   } else {
       PrintFormat("OrderSend SUCCESS for %s. Ticket: %d", symbol, res.order);
   }
   return ok && (res.retcode == TRADE_RETCODE_DONE || res.retcode == TRADE_RETCODE_PLACED);
}

long _gv_last_line()
{
   string key = "AG_CMD_LAST_LINE";
   if(!GlobalVariableCheck(key)) return 0;
   return (long)GlobalVariableGet(key);
}

void _gv_set_last_line(long v)
{
   GlobalVariableSet("AG_CMD_LAST_LINE", (double)v);
}

void _process_commands()
{
   int handle = FileOpen(_commands_file(), FILE_READ|FILE_CSV|FILE_COMMON);
   if(handle == INVALID_HANDLE) return;

   long last_line = _gv_last_line();
   long line_no = 0;

   string h_ts = FileReadString(handle);
   if(FileIsEnding(handle))
   {
      FileClose(handle);
      return;
   }
   string h_symbol = FileReadString(handle);
   string h_cmd = FileReadString(handle);
   string h_dir = FileReadString(handle);
   string h_entry = FileReadString(handle);
   string h_sl = FileReadString(handle);
   string h_tp = FileReadString(handle);
   string h_risk = FileReadString(handle);
   string h_magic = FileReadString(handle);
   string h_tol = FileReadString(handle);
   string h_maxpb = FileReadString(handle);
   string h_tag = FileReadString(handle);

   while(!FileIsEnding(handle))
   {
      string ts = FileReadString(handle);
      if(FileIsEnding(handle)) break;
      string symbol = FileReadString(handle);
      string cmd = FileReadString(handle);
      string dir_s = FileReadString(handle);
      string entry_s = FileReadString(handle);
      string sl_s = FileReadString(handle);
      string tp_s = FileReadString(handle);
      string risk_s = FileReadString(handle);
      string magic_s = FileReadString(handle);
      string tol_s = FileReadString(handle);
      string maxpb_s = FileReadString(handle);
      string tag_s = FileReadString(handle);

      line_no++;
      if(line_no <= last_line) continue;

      long magic = (long)StringToInteger(magic_s);
      if(cmd == "CANCEL_ALL")
      {
         _cancel_all_pending(symbol, magic);
      }
      else if(cmd == "PLACE_LIMIT")
      {
         int dir = (int)StringToInteger(dir_s);
         double entry = StringToDouble(entry_s);
         double sl = StringToDouble(sl_s);
         double tp = StringToDouble(tp_s);
         double risk = StringToDouble(risk_s);
         double tol_points = StringToDouble(tol_s);
         PrintFormat("Processing PLACE_LIMIT for %s dir=%d entry=%.5f sl=%.5f tp=%.5f", symbol, dir, entry, sl, tp);
         _place_limit(symbol, magic, dir, entry, sl, tp, risk, tol_points);
      }
   }

   FileClose(handle);
   if(line_no > last_line) _gv_set_last_line(line_no);
}

int OnInit()
{
   if(!_split_symbols(InpSymbols, _symbols)) return INIT_FAILED;
   EventSetTimer(InpTimerSec);
   for(int i=0;i<ArraySize(_symbols);i++)
      SymbolSelect(_symbols[i], true);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   for(int i=0;i<ArraySize(_symbols);i++)
   {
      string symbol = _symbols[i];
      _write_rates_csv(symbol, InpAnchorTF, InpExportBars);
      _write_rates_csv(symbol, InpExecutionTF, InpExportBars);
   }
   _process_commands();
}
