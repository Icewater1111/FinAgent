#该文件是金融实体库（Financial Entity Library, FEL）的核心模块，定义了实体类及其与 SQLite 数据库的交互逻辑。
import sqlite3
from typing import Optional, List, Dict, Any, Type, Tuple
import os

# --- 1. 基础类：FELBase ---
# 所有实体类的基类，提供通用的数据库操作和表结构定义
class FELBase:
    _db_manager: 'FELManager' = None # 用于持有数据库管理器实例
    _registered_entity_classes: List[Type['FELBase']] = [] # 注册所有继承FELBase的实体类

    def __init__(self, id: Optional[int] = None):
        self.id = id

    @classmethod
    def _get_table_name(cls) -> str:
        """根据类名获取对应的数据库表名（通常是类名的小写复数形式）。"""
        return cls.__name__.lower() + 's'

    @classmethod
    def _create_table_sql(cls) -> str:
        """子类必须实现此方法，返回创建对应表的 SQL 语句。"""
        raise NotImplementedError("Subclasses must implement _create_table_sql method.")

    def to_dict(self) -> Dict[str, Any]:
        """将实例转换为字典，用于数据库插入/更新。"""
        # 过滤掉 id 字段，因为它通常是自增的
        return {k: v for k, v in self.__dict__.items() if k != 'id'}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FELBase':
        """从字典创建实例，用于从数据库读取数据。"""
        instance = cls(**data)
        return instance

    def __repr__(self) -> str:
        attrs = ', '.join(f"{k}={getattr(self, k)!r}" for k in self.__dict__ if k != 'id')
        return f"{self.__class__.__name__}(id={self.id!r}, {attrs})"

    @classmethod
    def _register_entity(cls, entity_class: Type['FELBase']):
        """注册实体类，以便 FELManager 知道要创建哪些表。"""
        if entity_class not in cls._registered_entity_classes:
            cls._registered_entity_classes.append(entity_class)

# --- 2. 实体类定义 ---

class Company(FELBase):
    # 公司实体类
    def __init__(self, ticker: str, full_name_zh: str, short_name_zh: str,
                 market_id: Optional[int] = None, industry_id: Optional[int] = None,
                 full_name_en: Optional[str] = None, short_name_en: Optional[str] = None,
                 description_zh: Optional[str] = None, website: Optional[str] = None,
                 id: Optional[int] = None):
        super().__init__(id)
        self.ticker = ticker # 股票代码
        self.full_name_zh = full_name_zh # 中文全称
        self.short_name_zh = short_name_zh # 中文简称
        self.full_name_en = full_name_en # 英文全称
        self.short_name_en = short_name_en # 英文简称
        self.market_id = market_id # 所属市场ID
        self.industry_id = industry_id # 所属行业ID
        self.description_zh = description_zh # 中文描述
        self.website = website # 公司官网
    
    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            full_name_zh TEXT NOT NULL,
            short_name_zh TEXT NOT NULL,
            full_name_en TEXT,
            short_name_en TEXT,
            market_id INTEGER,
            industry_id INTEGER,
            description_zh TEXT,
            website TEXT,
            FOREIGN KEY (market_id) REFERENCES markets(id) ON DELETE SET NULL,
            FOREIGN KEY (industry_id) REFERENCES industries(id) ON DELETE SET NULL
        );
        """
FELBase._register_entity(Company)


class CompanyAlias(FELBase):
    # 公司别名实体类
    def __init__(self, company_id: int, alias_name: str, source: Optional[str] = None,
                 id: Optional[int] = None):
        super().__init__(id)
        self.company_id = company_id # 关联的公司ID
        self.alias_name = alias_name # 别名
        self.source = source # 别名来源

    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            alias_name TEXT NOT NULL,
            source TEXT,
            UNIQUE (company_id, alias_name), -- 同一家公司不能有重复别名
            FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
        );
        """
FELBase._register_entity(CompanyAlias)


class FinancialMetric(FELBase):
    # 金融指标实体类
    def __init__(self, standard_name_en: str, display_name_zh: str,
                 display_name_en: Optional[str] = None, definition_zh: Optional[str] = None,
                 definition_en: Optional[str] = None, unit: Optional[str] = None,
                 data_type: Optional[str] = None, granularity: Optional[str] = None,
                 category: Optional[str] = None, id: Optional[int] = None):
        super().__init__(id)
        self.standard_name_en = standard_name_en # 标准英文名称
        self.display_name_zh = display_name_zh # 中文显示名称
        self.display_name_en = display_name_en # 英文显示名称
        self.definition_zh = definition_zh # 中文定义
        self.definition_en = definition_en # 英文定义
        self.unit = unit # 计量单位
        self.data_type = data_type # 数据类型
        self.granularity = granularity # 数据粒度
        self.category = category # 指标类别

    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            standard_name_en TEXT UNIQUE NOT NULL,
            display_name_zh TEXT NOT NULL,
            display_name_en TEXT,
            definition_zh TEXT,
            definition_en TEXT,
            unit TEXT,
            data_type TEXT,
            granularity TEXT,
            category TEXT
        );
        """
FELBase._register_entity(FinancialMetric)


class Market(FELBase):
    # 市场实体类
    def __init__(self, name_zh: str, code: str, name_en: Optional[str] = None,
                 country_zh: Optional[str] = None, country_en: Optional[str] = None,
                 timezone: Optional[str] = None, id: Optional[int] = None):
        super().__init__(id)
        self.name_zh = name_zh # 中文名称
        self.name_en = name_en # 英文名称
        self.code = code # 市场代码
        self.country_zh = country_zh # 中文国家/地区
        self.country_en = country_en # 英文国家/地区
        self.timezone = timezone # 时区

    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_zh TEXT NOT NULL,
            name_en TEXT,
            code TEXT UNIQUE NOT NULL,
            country_zh TEXT,
            country_en TEXT,
            timezone TEXT
        );
        """
FELBase._register_entity(Market)


class Industry(FELBase):
    # 行业实体类
    def __init__(self, name_zh: str, name_en: Optional[str] = None,
                 classification_system: Optional[str] = None, parent_id: Optional[int] = None,
                 id: Optional[int] = None):
        super().__init__(id)
        self.name_zh = name_zh # 中文名称
        self.name_en = name_en # 英文名称
        self.classification_system = classification_system # 行业分类体系
        self.parent_id = parent_id # 上级行业ID

    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_zh TEXT UNIQUE NOT NULL,
            name_en TEXT,
            classification_system TEXT,
            parent_id INTEGER,
            FOREIGN KEY (parent_id) REFERENCES industries(id) ON DELETE SET NULL
        );
        """
FELBase._register_entity(Industry)


class Currency(FELBase):
    # 货币实体类
    def __init__(self, name_zh: str, code: str, name_en: Optional[str] = None,
                 symbol: Optional[str] = None, id: Optional[int] = None):
        super().__init__(id)
        self.name_zh = name_zh # 中文名称
        self.name_en = name_en # 英文名称
        self.code = code # 货币代码
        self.symbol = symbol # 货币符号

    @classmethod
    def _create_table_sql(cls) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {cls._get_table_name()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_zh TEXT NOT NULL,
            name_en TEXT,
            code TEXT UNIQUE NOT NULL,
            symbol TEXT
        );
        """
FELBase._register_entity(Currency)


# --- 3. FELManager：数据库管理与操作接口 ---

class FELManager:
    def __init__(self, db_path: str = 'financial_entities.db'):
        self.db_path = db_path
        FELBase._db_manager = self # 将自身注册到基类，方便实体类访问
        # 确保数据库文件所在的目录存在
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)


    def _connect(self) -> sqlite3.Connection:
        """建立数据库连接。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # 使查询结果可以通过字典键访问
        return conn

    def _execute(self, sql: str, params: Tuple = (), commit: bool = False) -> Optional[List[sqlite3.Row]]:
        """执行 SQL 语句并返回结果。"""
        conn = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if commit:
                conn.commit()
                return cursor.lastrowid # 返回最后插入的ID
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()

    def create_tables(self):
        """创建所有注册的实体对应的数据库表。"""
        # 确保外键约束在创建时被启用
        self._execute("PRAGMA foreign_keys = ON;")
        for entity_class in FELBase._registered_entity_classes:
            sql = entity_class._create_table_sql()
            self._execute(sql, commit=True)
            print(f"Table '{entity_class._get_table_name()}' created or already exists.")

    def insert(self, entity: FELBase) -> Optional[int]:
        """插入一个实体对象到对应的表中，并返回其ID。"""
        table_name = entity._get_table_name()
        data = entity.to_dict()
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        last_id = self._execute(sql, tuple(data.values()), commit=True)
        if last_id is not None:
            entity.id = last_id # 更新实体对象的ID
        return last_id

    def get_by_id(self, entity_class: Type[FELBase], entity_id: int) -> Optional[FELBase]:
        """根据ID获取一个实体对象。"""
        table_name = entity_class._get_table_name()
        sql = f"SELECT * FROM {table_name} WHERE id = ?"
        row = self._execute(sql, (entity_id,))
        if row:
            return entity_class.from_dict(dict(row[0]))
        return None

    def get_all(self, entity_class: Type[FELBase]) -> List[FELBase]:
        """获取所有指定类型的实体对象。"""
        table_name = entity_class._get_table_name()
        sql = f"SELECT * FROM {table_name}"
        rows = self._execute(sql)
        return [entity_class.from_dict(dict(row)) for row in rows] if rows else []

    def find_one(self, entity_class: Type[FELBase], **kwargs) -> Optional[FELBase]:
        """根据条件查询一个实体对象。"""
        table_name = entity_class._get_table_name()
        conditions = " AND ".join([f"{k} = ?" for k in kwargs.keys()])
        sql = f"SELECT * FROM {table_name} WHERE {conditions} LIMIT 1"
        row = self._execute(sql, tuple(kwargs.values()))
        if row:
            return entity_class.from_dict(dict(row[0]))
        return None

    def find_many(self, entity_class: Type[FELBase], **kwargs) -> List[FELBase]:
        """根据条件查询多个实体对象。"""
        table_name = entity_class._get_table_name()
        conditions = " AND ".join([f"{k} = ?" for k in kwargs.keys()])
        sql = f"SELECT * FROM {table_name} WHERE {conditions}"
        rows = self._execute(sql, tuple(kwargs.values()))
        return [entity_class.from_dict(dict(row)) for row in rows] if rows else []

    def update(self, entity: FELBase) -> bool:
        """更新一个实体对象。"""
        if entity.id is None:
            print("Error: Cannot update entity without an ID.")
            return False
        
        table_name = entity._get_table_name()
        data = entity.to_dict()
        set_clauses = ', '.join([f"{k} = ?" for k in data.keys()])
        sql = f"UPDATE {table_name} SET {set_clauses} WHERE id = ?"
        params = tuple(data.values()) + (entity.id,)
        result = self._execute(sql, params, commit=True)
        return result is not None # If result is None, an error occurred

    def delete(self, entity: FELBase) -> bool:
        """删除一个实体对象。"""
        if entity.id is None:
            print("Error: Cannot delete entity without an ID.")
            return False
        
        table_name = entity._get_table_name()
        sql = f"DELETE FROM {table_name} WHERE id = ?"
        result = self._execute(sql, (entity.id,), commit=True)
        return result is not None

# --- 示例使用 ---
if __name__ == "__main__":
    db_file = 'fel_test.db'
    # 清理旧的数据库文件以便每次运行都是全新开始
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database file: {db_file}")

    fel_manager = FELManager(db_path=db_file)
    fel_manager.create_tables()

    print("\n--- 插入数据 ---")
    # 插入市场
    hkex = Market(name_zh="香港交易所", code="HK", country_zh="中国香港")
    fel_manager.insert(hkex)
    print(f"Inserted: {hkex}")

    nasdaq = Market(name_zh="纳斯达克", code="NASDAQ", name_en="NASDAQ Stock Market", country_en="USA")
    fel_manager.insert(nasdaq)
    print(f"Inserted: {nasdaq}")

    # 插入行业
    it_industry = Industry(name_zh="信息技术", classification_system="GICS")
    fel_manager.insert(it_industry)
    print(f"Inserted: {it_industry}")

    finance_industry = Industry(name_zh="金融", classification_system="GICS")
    fel_manager.insert(finance_industry)
    print(f"Inserted: {finance_industry}")

    # 插入公司
    tencent = Company(ticker="00700.HK", full_name_zh="腾讯控股有限公司", short_name_zh="腾讯控股",
                      market_id=hkex.id, industry_id=it_industry.id, website="www.tencent.com")
    fel_manager.insert(tencent)
    print(f"Inserted: {tencent}")

    apple = Company(ticker="AAPL", full_name_zh="苹果公司", short_name_zh="苹果",
                    full_name_en="Apple Inc.", short_name_en="Apple",
                    market_id=nasdaq.id, industry_id=it_industry.id)
    fel_manager.insert(apple)
    print(f"Inserted: {apple}")

    # 插入公司别名
    alias_tencent_1 = CompanyAlias(company_id=tencent.id, alias_name="鹅厂")
    fel_manager.insert(alias_tencent_1)
    print(f"Inserted: {alias_tencent_1}")

    alias_tencent_2 = CompanyAlias(company_id=tencent.id, alias_name="腾子")
    fel_manager.insert(alias_tencent_2)
    print(f"Inserted: {alias_tencent_2}")

    # 插入金融指标
    pe_ratio = FinancialMetric(standard_name_en="P/E Ratio", display_name_zh="市盈率", unit="倍", category="估值指标")
    fel_manager.insert(pe_ratio)
    print(f"Inserted: {pe_ratio}")

    revenue = FinancialMetric(standard_name_en="Revenue", display_name_zh="营业收入", unit="元", granularity="年报", category="盈利能力")
    fel_manager.insert(revenue)
    print(f"Inserted: {revenue}")

    # 插入货币
    hkd = Currency(name_zh="港元", code="HKD", symbol="HK$")
    fel_manager.insert(hkd)
    print(f"Inserted: {hkd}")

    usd = Currency(name_zh="美元", code="USD", symbol="$")
    fel_manager.insert(usd)
    print(f"Inserted: {usd}")

    print("\n--- 查询数据 ---")
    # 获取所有公司
    all_companies = fel_manager.get_all(Company)
    print("\nAll Companies:")
    for company in all_companies:
        print(company)
        # 尝试获取关联的市场和行业信息
        market = fel_manager.get_by_id(Market, company.market_id) if company.market_id else None
        industry = fel_manager.get_by_id(Industry, company.industry_id) if company.industry_id else None
        print(f"  Market: {market.name_zh if market else 'N/A'}, Industry: {industry.name_zh if industry else 'N/A'}")

    # 根据 ticker 查找公司
    found_tencent = fel_manager.find_one(Company, ticker="00700.HK")
    print(f"\nFound Tencent by ticker: {found_tencent}")

    # 查找腾讯的别名
    tencent_aliases = fel_manager.find_many(CompanyAlias, company_id=tencent.id)
    print(f"\nTencent Aliases:")
    for alias in tencent_aliases:
        print(alias)

    # 获取所有金融指标
    all_metrics = fel_manager.get_all(FinancialMetric)
    print("\nAll Financial Metrics:")
    for metric in all_metrics:
        print(metric)

    print("\n--- 更新数据 ---")
    # 更新腾讯的描述
    if tencent:
        tencent.description_zh = "中国领先的互联网科技公司。"
        fel_manager.update(tencent)
        updated_tencent = fel_manager.get_by_id(Company, tencent.id)
        print(f"Updated Tencent: {updated_tencent}")

    print("\n--- 删除数据 ---")
    # 删除一个别名
    if alias_tencent_2:
        fel_manager.delete(alias_tencent_2)
        remaining_aliases = fel_manager.find_many(CompanyAlias, company_id=tencent.id)
        print(f"Remaining Tencent Aliases after deletion: {remaining_aliases}")

    # 删除腾讯公司，由于设置了 ON DELETE CASCADE，其别名也会被删除
    if tencent:
        fel_manager.delete(tencent)
        remaining_companies = fel_manager.get_all(Company)
        print(f"Remaining companies after deleting Tencent: {remaining_companies}")
        # 验证腾讯的别名是否也被删除
        tencent_aliases_after_company_delete = fel_manager.find_many(CompanyAlias, company_id=tencent.id)
        print(f"Tencent aliases after company deletion (should be empty): {tencent_aliases_after_company_delete}")

    print("\n--- 数据库操作完成 ---")
