SET max_bytes_before_external_group_by = 2000000000;

SELECT
    sdss_name,
    segue_tags,
    arr.1 AS ztf_oid,
    arr.2 AS filter,
    arr.3 AS mjd,
    arr.4 AS mag,
    arr.5 AS magerr
FROM
(
    SELECT
        sdss_name,
        any(object_tags) AS segue_tags,
        arraySort((arr) -> arr.3, groupArray((ztf_oid, filter, mjd, mag, magerr))) AS arr
    FROM
    (
        SELECT
            oid AS ztf_oid,
            filter,
            mjd,
            mag,
            magerr,
            sdss_name,
            object_tags
        FROM ztf.dr3
        INNER JOIN
        (
            SELECT
                sdss_name,
                object_tags,
                arrayJoin(h3kRing(h3index10, toUInt8(ceil((1. / 3600.) / h3EdgeAngle(10))))) AS h3index10,
                radeg AS ra_asassn,
                dedeg AS dec_asassn
            FROM sdss.stripe82_candidates
        ) AS asassn USING (h3index10)
        WHERE (catflags = 0) AND (greatCircleAngle(ra, dec, ra_asassn, dec_asassn) <= (1. / 3600.))
    )
    GROUP BY sdss_name
)
INTO OUTFILE '/tmp/ztf-sdss.csv'
FORMAT CSV
