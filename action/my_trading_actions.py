from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any

# 用umfutures这个库初始化client
# umfutures这个库主要是做合约下单


class TradingActions:
    def __init__(self, client, symbol):
        self.symbol_info_map = {}
        self.client = client
        self.symbol = symbol

    def initialize(self) -> None:
        """初始化交易对信息"""
        exchange_info = self.client.exchange_info()
        self.symbol_info_map = {item["symbol"]: item for item in exchange_info["symbols"]}

    def get_symbol_trade_info(self) -> None:
        """获取交易对信息"""
        item = self.symbol_info_map[self.symbol]
        self.quantity_precision = item["quantityPrecision"]
        self.min_notional = None
        print(f'下单精度为{self.quantity_precision},最小下单金额为{self.min_notional}')

        for f in item["filters"]:
            if f["filterType"] == "MIN_NOTIONAL":
                self.min_notional = float(f["notional"])  # 将可能的字符串转换为 float

        if self.min_notional is None:
            raise ValueError(f"MIN_NOTIONAL filter not found for symbol {self.symbol}")
    
    def get_account_info(self) -> Dict:
        """获取账户"""
        response: Dict[str, Any] = self.client.account()
        available_balance: float = float(response.get("availableBalance", 0.0))
        print(f'当前可用余额为{available_balance}')
        return available_balance
    
    def get_position_info(self) -> Dict:
        """获取持仓信息"""
        response: Dict[str, Any] = self.client.account()
        positions = response.get("positions", [])
        position_info: Dict[str, Any] = {
                "symbol": self.symbol,
                "long": {"amt": 0.0, "usdt": 0.0, "margin": 0.0},
                "short": {"amt": 0.0, "usdt": 0.0, "margin": 0.0},
                "direction": "none"
            }

        for pos in positions:
            if pos.get("symbol") != self.symbol:
                print(f'账户并无持仓{self.symbol}!')
                continue

            position_amt: float = float(pos.get("positionAmt"))
            entry_price: float = float(pos.get("entryPrice"))
            unrealized_profit: float = float(pos.get("unrealizedProfit"))

            # 防止 entry_price 为 0 导致计算错误
            notional_usdt: float = round(position_amt * entry_price, 4) if entry_price != 0 else 0.0
            margin: float = round(notional_usdt + unrealized_profit, 4)

            if position_amt > 0:
                position_info["long"] = {
                    "amt": position_amt,
                    "usdt": notional_usdt,
                    "margin": margin
                    }
                position_info["direction"] = "long"
            elif position_amt < 0:
                position_info["short"] = {
                    "amt": position_amt,
                    "usdt": notional_usdt,
                    "margin": margin
                }
                position_info["direction"] = "short"

        position_info = {'position_info': position_info}
        return position_info
    
    def get_order_qty(self):
        """计算下单数量"""
        available_balance = Decimal(self.get_account_info())
        print(f'当前的余额为{available_balance}')
        current_price = Decimal(self.client.ticker_price(self.symbol))
        print(f'当前交易对的价格为{current_price}')
        qty = available_balance / current_price
        print(f'所需下单数量为{qty}')
        return qty

    def handle_qty_precision(self, qty: float) -> Decimal:
        """调整成交数量以符合币安的精度要求"""
        self.get_symbol_trade_info()
        qty_decimal = Decimal(str(qty)).normalize()  # 保证精度

        if qty_decimal == 0:
            return Decimal("0")

        # 构造精度格式
        precision_format = '1' if self.quantity_precision == 0 else f'1.{"0" * self.quantity_precision}'
        rounded_qty = qty_decimal.quantize(Decimal(precision_format), rounding=ROUND_HALF_UP)
        print(f'调整前的成交数量为{qty},调整后的成交数量为{rounded_qty}')
        return rounded_qty

    def close_long(self,qty):
        """平多仓"""
        if qty > 0:
            final_qty = self.handle_qty_precision(abs(qty), self.symbol)
            self.client.new_order(
                symbol=self.symbol,
                side="SELL",
                type="MARKET",
                quantity=final_qty
            )
            print(f"📉 平多 {final_qty} {self.symbol}")
        else:
            print("✅ 当前没有多仓，无需平仓")

    def close_short(self,qty):
        """平空仓"""
        if qty < 0:
            final_qty = self.handle_qty_precision(abs(qty), self.symbol)
            self.client.new_order(
                symbol=self.symbol,
                side="BUY",
                type="MARKET",
                quantity=final_qty
            )
            print(f"📈 平空 {final_qty} {self.symbol}")
        else:
            print("✅ 当前没有空仓，无需平仓")

    def open_long(self,qty):
        """开多仓"""
        final_qty = self.handle_qty_precision(qty, self.symbol)  # 默认下1个单位
        self.client.new_order(
                symbol=self.symbol,
                side="BUY",
                type="MARKET",
                quantity=final_qty
            )
        print(f"📈 开多 {final_qty} {self.symbol}")

    def open_short(self,qty):
        """开空仓"""

        final_qty = self.handle_qty_precision(qty, self.symbol)  # 默认下1个单位
        self.client.new_order(
                symbol=self.symbol,
                side="SELL",
                type="MARKET",
                quantity=final_qty
            )
        print(f"📉 开空 {final_qty} {self.symbol}")
